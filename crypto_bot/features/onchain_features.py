"""
On-chain feature engineering for crypto trading signals.

Transforms raw blockchain data (whale transfers, exchange flows, network
metrics) into numerical features suitable for ML models.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class OnChainFeatures:
    """Derive ML-ready features from on-chain blockchain data."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        logger.info("OnChainFeatures initialised")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_dataframe(records: List[Dict[str, Any]], time_col: str = "timestamp") -> pd.DataFrame:
        """Convert a list of dicts to a time-indexed DataFrame.

        Parameters
        ----------
        records : list[dict]
            Raw records, each containing at least a *time_col* key.
        time_col : str
            Name of the timestamp field.

        Returns
        -------
        pd.DataFrame
            DataFrame sorted and indexed by datetime.

        Raises
        ------
        ValueError
            If *records* is empty or *time_col* is missing.
        """
        if not records:
            raise ValueError("records list is empty")
        df = pd.DataFrame(records)
        if time_col not in df.columns:
            raise ValueError(f"Expected column '{time_col}' not found in records")
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).set_index(time_col)
        return df

    # ------------------------------------------------------------------
    # whale transfer features
    # ------------------------------------------------------------------
    def compute_whale_pressure(self, transfers: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute whale buying/selling pressure from large-value transfers.

        Expected keys per transfer dict:
            timestamp, amount, direction ("buy" | "sell")

        Parameters
        ----------
        transfers : list[dict]
            Raw whale transfer records.

        Returns
        -------
        pd.DataFrame
            Columns: whale_buy_pressure, whale_sell_pressure, net_whale_flow.
        """
        df = self._to_dataframe(transfers)

        for col in ("amount", "direction"):
            if col not in df.columns:
                raise ValueError(f"Whale transfers missing required column: {col}")

        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

        buy_mask = df["direction"].str.lower() == "buy"
        sell_mask = df["direction"].str.lower() == "sell"

        # Resample to daily buckets and apply rolling 7-day mean
        buy_daily = df.loc[buy_mask, "amount"].resample("1D").sum().fillna(0)
        sell_daily = df.loc[sell_mask, "amount"].resample("1D").sum().fillna(0)

        # Align indices
        all_days = buy_daily.index.union(sell_daily.index)
        buy_daily = buy_daily.reindex(all_days, fill_value=0)
        sell_daily = sell_daily.reindex(all_days, fill_value=0)

        whale_buy_pressure = buy_daily.rolling(window=7, min_periods=1).mean()
        whale_sell_pressure = sell_daily.rolling(window=7, min_periods=1).mean()
        net_whale_flow = whale_buy_pressure - whale_sell_pressure

        result = pd.DataFrame(
            {
                "whale_buy_pressure": whale_buy_pressure,
                "whale_sell_pressure": whale_sell_pressure,
                "net_whale_flow": net_whale_flow,
            }
        )
        logger.info("Whale pressure features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # exchange flow features
    # ------------------------------------------------------------------
    def compute_exchange_flow_features(self, flows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute exchange inflow/outflow features.

        Expected keys per flow dict:
            timestamp, inflow, outflow

        Parameters
        ----------
        flows : list[dict]
            Raw exchange flow records.

        Returns
        -------
        pd.DataFrame
            Columns: exchange_inflow_ma, exchange_outflow_ma, net_flow_change.
        """
        df = self._to_dataframe(flows)

        for col in ("inflow", "outflow"):
            if col not in df.columns:
                raise ValueError(f"Exchange flows missing required column: {col}")

        df["inflow"] = pd.to_numeric(df["inflow"], errors="coerce").fillna(0)
        df["outflow"] = pd.to_numeric(df["outflow"], errors="coerce").fillna(0)

        inflow_ma = df["inflow"].rolling(window=7, min_periods=1).mean()
        outflow_ma = df["outflow"].rolling(window=7, min_periods=1).mean()

        net_flow = df["inflow"] - df["outflow"]
        net_flow_change = net_flow.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

        result = pd.DataFrame(
            {
                "exchange_inflow_ma": inflow_ma,
                "exchange_outflow_ma": outflow_ma,
                "net_flow_change": net_flow_change,
            },
            index=df.index,
        )
        logger.info("Exchange flow features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # network activity features
    # ------------------------------------------------------------------
    def compute_network_activity(self, metrics: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute network health features.

        Expected keys per metric dict:
            timestamp, active_addresses, hash_rate, gas_fee

        Parameters
        ----------
        metrics : list[dict]
            Raw network metric records.

        Returns
        -------
        pd.DataFrame
            Columns: active_addr_change, hash_rate_change, gas_fee_zscore.
        """
        df = self._to_dataframe(metrics)

        for col in ("active_addresses", "hash_rate", "gas_fee"):
            if col not in df.columns:
                raise ValueError(f"Network metrics missing required column: {col}")

        df["active_addresses"] = pd.to_numeric(df["active_addresses"], errors="coerce").fillna(0)
        df["hash_rate"] = pd.to_numeric(df["hash_rate"], errors="coerce").fillna(0)
        df["gas_fee"] = pd.to_numeric(df["gas_fee"], errors="coerce").fillna(0)

        active_addr_change = df["active_addresses"].pct_change().replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

        hash_rate_change = df["hash_rate"].pct_change().replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

        # Z-score of gas fees over a 30-period rolling window
        gas_mean = df["gas_fee"].rolling(window=30, min_periods=1).mean()
        gas_std = df["gas_fee"].rolling(window=30, min_periods=1).std().replace(0, np.nan)
        gas_fee_zscore = ((df["gas_fee"] - gas_mean) / gas_std).fillna(0)

        result = pd.DataFrame(
            {
                "active_addr_change": active_addr_change,
                "hash_rate_change": hash_rate_change,
                "gas_fee_zscore": gas_fee_zscore,
            },
            index=df.index,
        )
        logger.info("Network activity features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # aggregate
    # ------------------------------------------------------------------
    def compute_all(
        self,
        transfers: List[Dict[str, Any]],
        flows: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Compute and merge all on-chain feature groups.

        Parameters
        ----------
        transfers : list[dict]
            Whale transfer records.
        flows : list[dict]
            Exchange flow records.
        metrics : list[dict]
            Network metric records.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with all on-chain features.
        """
        frames: List[pd.DataFrame] = []

        try:
            frames.append(self.compute_whale_pressure(transfers))
        except Exception as exc:
            logger.warning("Whale pressure computation failed: %s", exc)

        try:
            frames.append(self.compute_exchange_flow_features(flows))
        except Exception as exc:
            logger.warning("Exchange flow computation failed: %s", exc)

        try:
            frames.append(self.compute_network_activity(metrics))
        except Exception as exc:
            logger.warning("Network activity computation failed: %s", exc)

        if not frames:
            logger.error("All on-chain feature computations failed")
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        logger.info(
            "All on-chain features merged: %d columns, %d rows",
            result.shape[1],
            result.shape[0],
        )
        return result
