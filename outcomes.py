from __future__ import annotations

import bisect
import time
from typing import Any


HORIZONS_HOURS = (24, 48, 72)


def price_at_or_after(
    timestamps: list[int],
    prices: list[float],
    target_timestamp: int,
) -> float | None:
    if len(timestamps) != len(prices) or not timestamps:
        return None
    idx = bisect.bisect_left(timestamps, target_timestamp)
    if idx >= len(prices):
        return None
    return float(prices[idx])


def did_revert(price: float, target_price: float, direction: str) -> bool:
    if direction == "BUY_YES":
        return price >= target_price
    if direction == "BUY_NO":
        return price <= target_price
    return False


def evaluate_signal_outcome(
    signal: dict[str, Any],
    prices: list[float],
    timestamps: list[int],
) -> dict[str, Any]:
    target_price = signal.get("target_price")
    direction = signal.get("direction")
    run_timestamp = signal.get("run_timestamp")
    if target_price is None or direction is None or run_timestamp is None:
        return {}

    outcome: dict[str, Any] = {"outcome_checked_at": int(time.time() * 1000)}
    target_float = float(target_price)
    for horizon in HORIZONS_HOURS:
        price = price_at_or_after(
            timestamps,
            prices,
            int(run_timestamp) + horizon * 60 * 60 * 1000,
        )
        if price is None:
            continue
        outcome[f"price_{horizon}h"] = price
        outcome[f"reverted_{horizon}h"] = 1 if did_revert(price, target_float, str(direction)) else 0
    return outcome

