"""
On-chain / blockchain data collector.

Gathers whale transfers, active address counts, gas prices,
exchange flow metrics, and network-level statistics from Alchemy
and Glassnode-style HTTP APIs using async I/O.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import aiohttp

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------

class _TTLCache:
    """Minimalist in-memory cache with per-key time-to-live."""

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._store: Dict[str, tuple[float, Any]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        self._store[key] = (time.monotonic() + ttl, value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket style, async-friendly)
# ---------------------------------------------------------------------------

class _AsyncRateLimiter:
    """Simple token-bucket rate limiter for async code."""

    def __init__(self, calls_per_second: float = 5.0) -> None:
        self._min_interval = 1.0 / calls_per_second
        self._last_call: float = 0.0

    async def acquire(self) -> None:
        import asyncio

        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

# Default API base URLs
_ALCHEMY_BASE = "https://eth-mainnet.g.alchemy.com/v2"
_GLASSNODE_BASE = "https://api.glassnode.com/v1"
_ETHERSCAN_BASE = "https://api.etherscan.io/api"

# Cache TTL values (seconds)
_CACHE_SHORT = 120          # 2 min  -- gas fees, transfers
_CACHE_MEDIUM = 600         # 10 min -- active addresses, flows
_CACHE_LONG = 1800          # 30 min -- network metrics


class OnChainCollector:
    """Collect on-chain blockchain data from Alchemy and Glassnode APIs."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with API keys sourced from *settings* / environment.

        Expected environment variables (fallback when not on *settings*)::

            ALCHEMY_API_KEY
            GLASSNODE_API_KEY
            ETHERSCAN_API_KEY   (optional, used for whale transfers)
        """
        self._settings = settings

        self._alchemy_key: str = getattr(
            settings, "ALCHEMY_API_KEY", ""
        ) or os.environ.get("ALCHEMY_API_KEY", "")

        self._glassnode_key: str = getattr(
            settings, "GLASSNODE_API_KEY", ""
        ) or os.environ.get("GLASSNODE_API_KEY", "")

        self._etherscan_key: str = getattr(
            settings, "ETHERSCAN_API_KEY", ""
        ) or os.environ.get("ETHERSCAN_API_KEY", "")

        self._cache = _TTLCache(default_ttl=_CACHE_MEDIUM)
        self._limiter = _AsyncRateLimiter(calls_per_second=5.0)

        self._session: Optional[aiohttp.ClientSession] = None

        logger.info("OnChainCollector initialised")

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------ #
    # Generic request helper
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Perform a rate-limited GET request and return decoded JSON."""
        await self._limiter.acquire()
        session = await self._get_session()

        try:
            async with session.get(url, params=params, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientResponseError as exc:
            logger.error("HTTP %s for %s: %s", exc.status, url, exc.message)
            raise
        except aiohttp.ClientError as exc:
            logger.error("Request failed for %s: %s", url, exc)
            raise

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def fetch_whale_transfers(
        self,
        min_value_usd: float = 100_000,
    ) -> List[Dict[str, Any]]:
        """Return recent large-value transfers on Ethereum.

        Uses the Alchemy ``alchemy_getAssetTransfers`` JSON-RPC method.
        Falls back to Etherscan's token-transfer endpoint when an
        Etherscan key is available.

        Parameters
        ----------
        min_value_usd:
            Minimum USD-equivalent value to qualify as a *whale* transfer.
        """
        cache_key = f"whale_transfers:{min_value_usd}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        transfers: List[Dict[str, Any]] = []

        if self._alchemy_key:
            transfers = await self._fetch_whale_alchemy(min_value_usd)
        elif self._etherscan_key:
            transfers = await self._fetch_whale_etherscan(min_value_usd)
        else:
            logger.warning(
                "No Alchemy or Etherscan API key configured; "
                "whale transfer data unavailable"
            )

        self._cache.set(cache_key, transfers, ttl=_CACHE_SHORT)
        return transfers

    async def fetch_active_addresses(
        self,
        asset: str = "BTC",
    ) -> Dict[str, Any]:
        """Fetch the current active-address count and recent trend.

        Parameters
        ----------
        asset:
            Blockchain asset ticker (``"BTC"``, ``"ETH"``, etc.).

        Returns
        -------
        dict
            ``count`` -- latest active-address count.
            ``trend`` -- ``"increasing"``, ``"decreasing"``, or
            ``"stable"`` based on the last 7 data points.
        """
        cache_key = f"active_addresses:{asset}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result: Dict[str, Any] = {"asset": asset, "count": None, "trend": "unknown"}

        if not self._glassnode_key:
            logger.warning("Glassnode API key not set; active address data unavailable")
            return result

        url = f"{_GLASSNODE_BASE}/metrics/addresses/active_count"
        params = {
            "a": asset,
            "api_key": self._glassnode_key,
            "i": "24h",
            "s": "",  # let API default to recent window
        }

        try:
            data = await self._request(url, params=params)
            if data and isinstance(data, list) and len(data) > 0:
                result["count"] = data[-1].get("v")
                result["trend"] = self._compute_trend([p.get("v", 0) for p in data[-7:]])
        except Exception:
            logger.exception("Failed to fetch active addresses for %s", asset)

        self._cache.set(cache_key, result, ttl=_CACHE_MEDIUM)
        return result

    async def fetch_gas_fees(self) -> Dict[str, Any]:
        """Fetch current Ethereum gas prices.

        Returns
        -------
        dict
            ``slow``, ``standard``, ``fast`` -- gas prices in Gwei.
            ``base_fee`` -- current base fee (EIP-1559).
        """
        cache_key = "gas_fees"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result: Dict[str, Any] = {
            "slow": None,
            "standard": None,
            "fast": None,
            "base_fee": None,
        }

        # Try Alchemy first (eth_gasPrice + fee history)
        if self._alchemy_key:
            try:
                url = f"{_ALCHEMY_BASE}/{self._alchemy_key}"
                payload_gas = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_gasPrice",
                    "params": [],
                }
                payload_fee = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "eth_feeHistory",
                    "params": ["0x5", "latest", [25, 50, 75]],
                }

                session = await self._get_session()

                await self._limiter.acquire()
                async with session.post(url, json=payload_gas) as resp:
                    resp.raise_for_status()
                    gas_data = await resp.json()

                await self._limiter.acquire()
                async with session.post(url, json=payload_fee) as resp:
                    resp.raise_for_status()
                    fee_data = await resp.json()

                gas_price_wei = int(gas_data.get("result", "0x0"), 16)
                result["standard"] = round(gas_price_wei / 1e9, 2)

                fee_result = fee_data.get("result", {})
                base_fees = fee_result.get("baseFeePerGas", [])
                if base_fees:
                    result["base_fee"] = round(
                        int(base_fees[-1], 16) / 1e9, 2
                    )

                reward = fee_result.get("reward", [])
                if reward and reward[-1]:
                    percentiles = reward[-1]
                    result["slow"] = round(int(percentiles[0], 16) / 1e9, 2)
                    result["fast"] = round(int(percentiles[-1], 16) / 1e9, 2)

            except Exception:
                logger.exception("Failed to fetch gas fees via Alchemy")

        # Fallback: Etherscan gas oracle
        elif self._etherscan_key:
            try:
                params = {
                    "module": "gastracker",
                    "action": "gasoracle",
                    "apikey": self._etherscan_key,
                }
                data = await self._request(_ETHERSCAN_BASE, params=params)
                oracle = data.get("result", {})
                result["slow"] = float(oracle.get("SafeGasPrice", 0))
                result["standard"] = float(oracle.get("ProposeGasPrice", 0))
                result["fast"] = float(oracle.get("FastGasPrice", 0))
            except Exception:
                logger.exception("Failed to fetch gas fees via Etherscan")
        else:
            logger.warning("No API key for gas fee retrieval")

        self._cache.set(cache_key, result, ttl=_CACHE_SHORT)
        return result

    async def fetch_exchange_flows(
        self,
        asset: str = "ETH",
    ) -> Dict[str, Any]:
        """Fetch exchange inflow / outflow for *asset*.

        Returns
        -------
        dict
            ``inflow``, ``outflow``, ``net_flow`` -- 24h volumes.
        """
        cache_key = f"exchange_flows:{asset}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result: Dict[str, Any] = {
            "asset": asset,
            "inflow": None,
            "outflow": None,
            "net_flow": None,
        }

        if not self._glassnode_key:
            logger.warning(
                "Glassnode API key not set; exchange flow data unavailable"
            )
            return result

        try:
            base_params = {
                "a": asset,
                "api_key": self._glassnode_key,
                "i": "24h",
            }

            inflow_url = f"{_GLASSNODE_BASE}/metrics/transactions/transfers_volume_to_exchanges_sum"
            outflow_url = f"{_GLASSNODE_BASE}/metrics/transactions/transfers_volume_from_exchanges_sum"

            inflow_data = await self._request(inflow_url, params=base_params)
            outflow_data = await self._request(outflow_url, params=base_params)

            if inflow_data and isinstance(inflow_data, list) and inflow_data:
                result["inflow"] = inflow_data[-1].get("v")
            if outflow_data and isinstance(outflow_data, list) and outflow_data:
                result["outflow"] = outflow_data[-1].get("v")

            if result["inflow"] is not None and result["outflow"] is not None:
                result["net_flow"] = result["inflow"] - result["outflow"]
        except Exception:
            logger.exception("Failed to fetch exchange flows for %s", asset)

        self._cache.set(cache_key, result, ttl=_CACHE_MEDIUM)
        return result

    async def fetch_network_metrics(
        self,
        asset: str = "BTC",
    ) -> Dict[str, Any]:
        """Fetch network-level metrics for *asset*.

        Returns
        -------
        dict
            ``hash_rate``, ``difficulty``, ``block_height``,
            ``avg_block_time``, ``mempool_size``.
        """
        cache_key = f"network_metrics:{asset}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result: Dict[str, Any] = {
            "asset": asset,
            "hash_rate": None,
            "difficulty": None,
            "block_height": None,
            "avg_block_time": None,
            "mempool_size": None,
        }

        if not self._glassnode_key:
            logger.warning(
                "Glassnode API key not set; network metrics unavailable"
            )
            return result

        base_params = {
            "a": asset,
            "api_key": self._glassnode_key,
            "i": "24h",
        }

        metric_map = {
            "hash_rate": f"{_GLASSNODE_BASE}/metrics/mining/hash_rate_mean",
            "difficulty": f"{_GLASSNODE_BASE}/metrics/mining/difficulty_latest",
            "block_height": f"{_GLASSNODE_BASE}/metrics/blockchain/block_height",
        }

        for key, url in metric_map.items():
            try:
                data = await self._request(url, params=base_params)
                if data and isinstance(data, list) and data:
                    result[key] = data[-1].get("v")
            except Exception:
                logger.warning("Could not fetch %s for %s", key, asset)

        self._cache.set(cache_key, result, ttl=_CACHE_LONG)
        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    async def _fetch_whale_alchemy(
        self,
        min_value_usd: float,
    ) -> List[Dict[str, Any]]:
        """Fetch large ETH transfers via Alchemy's asset-transfer API."""
        url = f"{_ALCHEMY_BASE}/{self._alchemy_key}"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": "latest",
                    "toBlock": "latest",
                    "category": ["external", "erc20"],
                    "withMetadata": True,
                    "excludeZeroValue": True,
                    "maxCount": "0x64",
                    "order": "desc",
                }
            ],
        }

        transfers: List[Dict[str, Any]] = []
        try:
            session = await self._get_session()
            await self._limiter.acquire()
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

            raw_transfers = (
                data.get("result", {}).get("transfers", [])
            )

            for tx in raw_transfers:
                value = tx.get("value") or 0
                # Rough USD filter (caller should refine with price feed)
                if value >= min_value_usd / 3000:  # ~ETH price estimate
                    transfers.append(
                        {
                            "hash": tx.get("hash"),
                            "from": tx.get("from"),
                            "to": tx.get("to"),
                            "value": value,
                            "asset": tx.get("asset"),
                            "category": tx.get("category"),
                            "block_num": tx.get("blockNum"),
                            "metadata": tx.get("metadata"),
                        }
                    )
        except Exception:
            logger.exception("Alchemy whale transfer fetch failed")

        return transfers

    async def _fetch_whale_etherscan(
        self,
        min_value_usd: float,
    ) -> List[Dict[str, Any]]:
        """Fetch large ETH transactions via Etherscan."""
        params = {
            "module": "account",
            "action": "txlist",
            "address": "0x0000000000000000000000000000000000000000",
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 100,
            "sort": "desc",
            "apikey": self._etherscan_key,
        }

        transfers: List[Dict[str, Any]] = []
        try:
            data = await self._request(_ETHERSCAN_BASE, params=params)
            for tx in data.get("result", []):
                value_eth = int(tx.get("value", "0")) / 1e18
                if value_eth >= min_value_usd / 3000:
                    transfers.append(
                        {
                            "hash": tx.get("hash"),
                            "from": tx.get("from"),
                            "to": tx.get("to"),
                            "value": value_eth,
                            "asset": "ETH",
                            "block_num": tx.get("blockNumber"),
                            "timestamp": tx.get("timeStamp"),
                        }
                    )
        except Exception:
            logger.exception("Etherscan whale transfer fetch failed")

        return transfers

    @staticmethod
    def _compute_trend(values: List[float]) -> str:
        """Determine a simple trend direction from a short series."""
        if len(values) < 2:
            return "unknown"

        valid = [v for v in values if v is not None and v > 0]
        if len(valid) < 2:
            return "unknown"

        first_half = sum(valid[: len(valid) // 2]) / max(len(valid) // 2, 1)
        second_half = sum(valid[len(valid) // 2 :]) / max(
            len(valid) - len(valid) // 2, 1
        )

        pct_change = (second_half - first_half) / first_half if first_half else 0

        if pct_change > 0.03:
            return "increasing"
        if pct_change < -0.03:
            return "decreasing"
        return "stable"
