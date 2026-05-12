"""
Central asset router — dispatches UnifiedSignals to the correct lane executor.

Reads ``AssetSettings.enabled_assets`` and initialises only the active lanes.
CRYPTO reuses the existing TradeExecutor, STOCK and BET use their respective
executors. Missing API keys → graceful degradation to paper-only for that lane.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.config.asset_types import AssetType, UnifiedSignal
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class AssetRouter:
    """Top-level dispatcher for multi-asset trading.

    Routes every :class:`UnifiedSignal` to the correct lane executor based on
    ``signal.asset_type``. Aggregates balance and positions across all active
    lanes into a unified report.

    Args:
        settings: Bot configuration.
        risk_manager: Shared :class:`RiskManager` instance.
        mode: ``"paper"`` or ``"live"``.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        risk_manager: Optional[Any] = None,
        mode: str = "paper",
    ) -> None:
        self._settings = settings or _default_settings
        self._mode = mode
        self._risk_manager = risk_manager
        self._enabled: List[AssetType] = []
        self._executors: Dict[AssetType, Any] = {}
        self._lane_status: Dict[AssetType, str] = {}

        self._init_lanes()

    # ------------------------------------------------------------------
    # Lane initialisation
    # ------------------------------------------------------------------

    def _init_lanes(self) -> None:
        enabled_names = self._settings.asset.enabled_assets

        for name in enabled_names:
            try:
                asset_type = AssetType(name)
            except ValueError:
                logger.warning("Unknown asset type in config: %s — skipping", name)
                continue

            if asset_type == AssetType.CRYPTO:
                self._init_crypto_lane()
            elif asset_type == AssetType.STOCK:
                self._init_stock_lane()
            elif asset_type == AssetType.BET:
                self._init_bet_lane()

        logger.info(
            "AssetRouter initialised — enabled: %s",
            [t.value for t in self._enabled],
        )

    def _init_crypto_lane(self) -> None:
        try:
            from omnitrade.execution.trade_executor import TradeExecutor

            if self._risk_manager is None:
                from omnitrade.risk.risk_manager import RiskManager
                rm = RiskManager(self._settings)
            else:
                rm = self._risk_manager

            executor = TradeExecutor(risk_manager=rm, settings=self._settings)
            self._executors[AssetType.CRYPTO] = executor
            self._enabled.append(AssetType.CRYPTO)
            self._lane_status[AssetType.CRYPTO] = "active"
            logger.info("CRYPTO lane active (%s mode)", self._mode)
        except Exception as exc:
            logger.error("Failed to init CRYPTO lane: %s", exc)
            self._lane_status[AssetType.CRYPTO] = f"error: {exc}"

    def _init_stock_lane(self) -> None:
        try:
            from omnitrade.execution.stock_executor import StockExecutor

            if self._risk_manager is None:
                from omnitrade.risk.risk_manager import RiskManager
                rm = RiskManager(self._settings)
            else:
                rm = self._risk_manager

            executor = StockExecutor(risk_manager=rm, settings=self._settings)
            self._executors[AssetType.STOCK] = executor
            self._enabled.append(AssetType.STOCK)
            self._lane_status[AssetType.STOCK] = "active"
            logger.info("STOCK lane active (%s mode)", "paper" if executor.paper_mode else "live")
        except Exception as exc:
            logger.error("Failed to init STOCK lane: %s", exc)
            self._lane_status[AssetType.STOCK] = f"error: {exc}"

    def _init_bet_lane(self) -> None:
        try:
            from omnitrade.execution.betting_executor import BettingExecutor
            from omnitrade.risk.betting_risk import BettingRiskManager

            rm = BettingRiskManager(self._settings)
            executor = BettingExecutor(settings=self._settings, risk_manager=rm)
            self._executors[AssetType.BET] = executor
            self._enabled.append(AssetType.BET)
            self._lane_status[AssetType.BET] = "active"
            logger.info("BET lane active (paper mode, bankroll=$%.2f)", executor.get_bankroll())
        except Exception as exc:
            logger.error("Failed to init BET lane: %s", exc)
            self._lane_status[AssetType.BET] = f"error: {exc}"

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_signal(self, signal: UnifiedSignal) -> Dict:
        """Dispatch a unified signal to the correct executor.

        Args:
            signal: :class:`UnifiedSignal` with ``asset_type`` set.

        Returns:
            Execution result dict (same format as TradeExecutor).
        """
        if signal.asset_type not in self._executors:
            logger.warning(
                "No executor for %s — signal dropped (%s %s)",
                signal.asset_type.value,
                signal.symbol,
                signal.side,
            )
            return {
                "order_id": None,
                "status": "dropped",
                "reason": f"No executor for asset type {signal.asset_type.value}",
                "symbol": signal.symbol,
            }

        executor = self._executors[signal.asset_type]

        if signal.asset_type == AssetType.CRYPTO:
            return self._route_crypto(signal, executor)
        elif signal.asset_type == AssetType.STOCK:
            return executor.execute_trade(signal)
        elif signal.asset_type == AssetType.BET:
            return executor.execute_trade(signal)

        return {"order_id": None, "status": "unknown_asset_type", "symbol": signal.symbol}

    def _route_crypto(self, signal: UnifiedSignal, executor: Any) -> Dict:
        trade_signal = {
            "symbol": signal.symbol,
            "side": signal.side.lower(),
            "amount": signal.amount,
        }
        return executor.execute_trade(trade_signal)

    # ------------------------------------------------------------------
    # Unified reporting
    # ------------------------------------------------------------------

    def get_unified_balance(self) -> Dict[str, Any]:
        """Aggregate balances across all active lanes."""
        total_usd = 0.0
        lanes: Dict[str, Any] = {}

        for asset_type, executor in self._executors.items():
            try:
                bal = executor.get_balance()
                lanes[asset_type.value] = bal
                for currency, amounts in bal.items():
                    if "total" in amounts:
                        total_usd += amounts["total"]
            except Exception as exc:
                logger.warning("Balance fetch failed for %s: %s", asset_type.value, exc)
                lanes[asset_type.value] = {"error": str(exc)}

        return {"total_usd_equivalent": round(total_usd, 2), "by_lane": lanes}

    def get_all_positions(self) -> List[Dict]:
        """Aggregate open positions across all lanes."""
        all_positions: List[Dict] = []
        for asset_type, executor in self._executors.items():
            try:
                positions = executor.get_positions()
                for p in positions:
                    p["asset_type"] = asset_type.value
                all_positions.extend(positions)
            except Exception as exc:
                logger.warning("Position fetch failed for %s: %s", asset_type.value, exc)
        return all_positions

    def close_all_positions(self) -> List[Dict]:
        """Emergency liquidation across all lanes."""
        results: List[Dict] = []
        for asset_type, executor in self._executors.items():
            try:
                lane_results = executor.close_all_positions()
                for r in lane_results:
                    r["asset_type"] = asset_type.value
                results.extend(lane_results)
            except Exception as exc:
                logger.error("Close-all failed for %s: %s", asset_type.value, exc)
                results.append({"asset_type": asset_type.value, "status": "error", "reason": str(exc)})
        return results

    def get_status(self) -> Dict[str, Any]:
        """Return status of each lane."""
        enabled_count = len(self._enabled)
        return {
            "enabled_lanes": [t.value for t in self._enabled],
            "active_count": enabled_count,
            "lane_status": self._lane_status,
            "has_crypto": AssetType.CRYPTO in self._executors,
            "has_stock": AssetType.STOCK in self._executors,
            "has_bet": AssetType.BET in self._executors,
        }

    @property
    def enabled_asset_types(self) -> List[AssetType]:
        return list(self._enabled)

    @property
    def crypto_executor(self) -> Optional[Any]:
        return self._executors.get(AssetType.CRYPTO)

    @property
    def stock_executor(self) -> Optional[Any]:
        return self._executors.get(AssetType.STOCK)

    @property
    def bet_executor(self) -> Optional[Any]:
        return self._executors.get(AssetType.BET)
