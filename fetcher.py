from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import Settings, settings


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _field(market: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in market and market[name] is not None:
            return market[name]
    return default


def _is_active_market(market: dict[str, Any]) -> bool:
    if "active" in market:
        return bool(market["active"])
    if "closed" in market:
        return not bool(market["closed"])
    return True


def _market_volume_24h(market: dict[str, Any]) -> float:
    return _as_float(
        _field(
            market,
            "volume_24h",
            "volume24h",
            "volume24hr",
            "volume_24hr",
            "volumeNum",
            "volume",
            default=0,
        )
    )


def _market_end_date(market: dict[str, Any]) -> datetime | None:
    return _parse_datetime(
        _field(market, "end_date", "endDate", "endDateIso", "end_date_iso", "end")
    )


def _market_category(market: dict[str, Any]) -> str:
    category = _field(market, "category", "groupItemTitle", default=None)
    if isinstance(category, dict):
        category = category.get("name")
    if category:
        return str(category).strip().lower()
    tags = market.get("tags")
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, dict):
            return str(first.get("label") or first.get("name") or "uncategorized").lower()
        return str(first).lower()
    return "uncategorized"


def _market_condition_id(market: dict[str, Any]) -> str | None:
    value = _field(market, "condition_id", "conditionId", "conditionID", "id")
    return str(value) if value else None


def _market_history_ids(market: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    condition_id = _market_condition_id(market)
    if condition_id:
        ids.append(condition_id)

    tokens = market.get("tokens") or market.get("outcomes") or []
    if isinstance(tokens, list):
        for token in tokens:
            if isinstance(token, dict):
                token_id = token.get("token_id") or token.get("tokenId") or token.get("id")
                if token_id:
                    ids.append(str(token_id))

    return list(dict.fromkeys(ids))


def _normalize_history(payload: dict[str, Any] | list[Any]) -> tuple[list[float], list[int]]:
    rows: list[Any]
    if isinstance(payload, dict):
        rows = payload.get("history") or payload.get("data") or payload.get("prices") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    points: list[tuple[int, float]] = []
    for row in rows:
        if isinstance(row, dict):
            timestamp = row.get("t") or row.get("timestamp") or row.get("time")
            price = row.get("p") or row.get("price") or row.get("close")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            timestamp, price = row[0], row[1]
        else:
            continue

        if timestamp is None or price is None:
            continue
        ts = int(float(timestamp))
        if ts < 10_000_000_000:
            ts *= 1000
        points.append((ts, _as_float(price)))

    points.sort(key=lambda point: point[0])
    timestamps = [point[0] for point in points]
    prices = [point[1] for point in points]
    return prices, timestamps


class PolymarketFetcher:
    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self._semaphore = asyncio.Semaphore(cfg.request_concurrency)

    def _get_sync(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.cfg.polymarket_clob_url}{path}{query}"
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "fft-signal-agent/1.0"},
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urlopen(request, timeout=self.cfg.request_timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, ConnectionResetError) as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(min(0.5 * (2**attempt), 8))
        raise RuntimeError(f"GET failed for {url}: {last_error}")

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        async with self._semaphore:
            await asyncio.sleep(self.cfg.request_spacing_seconds)
            return await asyncio.to_thread(self._get_sync, path, params)

    async def fetch_markets(self) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"closed": "false"}
            if cursor:
                params["next_cursor"] = cursor
            if self.cfg.max_markets:
                params["limit"] = min(self.cfg.max_markets, 500)
            payload = await self._get("/markets", params=params)
            page = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(page, list):
                break
            markets.extend(market for market in page if isinstance(market, dict))

            if self.cfg.max_markets and len(markets) >= self.cfg.max_markets:
                return markets[: self.cfg.max_markets]

            cursor = None
            if isinstance(payload, dict):
                next_cursor = payload.get("next_cursor") or payload.get("nextCursor")
                if next_cursor and next_cursor != "LTE=":
                    cursor = str(next_cursor)
            if not cursor:
                break
        return markets

    async def fetch_price_history(self, condition_id: str) -> tuple[list[float], list[int]]:
        payload = await self._get(
            "/prices-history",
            params={"market": condition_id, "interval": "1h", "fidelity": 60},
        )
        return _normalize_history(payload)

    def market_passes_static_filters(self, market: dict[str, Any]) -> bool:
        if not _is_active_market(market):
            return False
        if _market_volume_24h(market) <= self.cfg.min_volume_24h:
            return False
        end_date = _market_end_date(market)
        if end_date:
            hours_to_resolution = (end_date - datetime.now(timezone.utc)).total_seconds() / 3600
            if hours_to_resolution <= self.cfg.resolution_buffer_hours:
                return False
        return _market_condition_id(market) is not None

    async def fetch_active_contracts(self) -> list[dict[str, Any]]:
        markets = [m for m in await self.fetch_markets() if self.market_passes_static_filters(m)]
        if self.cfg.max_markets:
            markets = markets[: self.cfg.max_markets]

        async def build_contract(market: dict[str, Any]) -> dict[str, Any] | None:
            condition_id = _market_condition_id(market)
            if not condition_id:
                return None
            prices: list[float] = []
            timestamps: list[int] = []
            for history_id in _market_history_ids(market):
                try:
                    prices, timestamps = await self.fetch_price_history(history_id)
                except (HTTPError, URLError, TimeoutError, ConnectionResetError):
                    continue
                if len(prices) >= self.cfg.min_price_history_hours:
                    break
            if len(prices) < self.cfg.min_price_history_hours:
                return None
            return {
                "condition_id": condition_id,
                "question": str(_field(market, "question", "title", default="Unknown contract")),
                "category": _market_category(market),
                "end_date": _market_end_date(market),
                "volume_24h": _market_volume_24h(market),
                "prices": prices,
                "timestamps": timestamps,
                "market_url": _field(market, "market_url", "url", "slug", default=None),
            }

        contracts = await asyncio.gather(*(build_contract(market) for market in markets))
        return [contract for contract in contracts if contract is not None]


async def fetch_active_contracts() -> list[dict[str, Any]]:
    return await PolymarketFetcher().fetch_active_contracts()
