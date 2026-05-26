from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from classifier import classify_signal
from config import settings
from fetcher import (
    PolymarketFetcher,
    _market_category,
    _market_condition_id,
    _market_history_ids,
    _market_volume_24h,
    _parse_datetime,
)
from fft_engine import (
    estimate_reversion_target,
    reconstruct_cycle_price,
    resample_price_history,
    run_fft_analysis,
)
from ml import FEATURE_COLUMNS, TARGET_COLUMN


def determine_direction_simple(window: list[float], features: dict[str, Any]) -> str | None:
    _target_price, direction = estimate_reversion_target(window, features)
    return direction


def generate_training_samples(contract: dict[str, Any]) -> list[dict[str, Any]]:
    prices = contract["prices"]
    samples: list[dict[str, Any]] = []

    step_hours = max(1, settings.run_interval_minutes // 60)
    for end_idx in range(settings.min_price_history_hours, len(prices), step_hours):
        window = prices[:end_idx]
        features = run_fft_analysis(window)
        signal_class = classify_signal(features)

        if signal_class != "ANOMALY":
            continue

        current_price = window[-1]
        direction = determine_direction_simple(window, features)
        future_window = prices[end_idx : end_idx + 72]
        if len(future_window) < 24 or direction is None:
            continue

        reconstructed = reconstruct_cycle_price(window, float(features["dominant_freq"]))
        target_price = float(reconstructed[-1])

        if direction == "BUY_YES":
            reverted = any(price >= target_price for price in future_window)
        else:
            reverted = any(price <= target_price for price in future_window)

        samples.append(
            {
                **features,
                "category": contract["category"],
                "volume": contract["volume_24h"],
                "duration_days": contract["duration_days"],
                "current_price": current_price,
                "target_price": target_price,
                "direction": direction,
                "label": 1 if reverted else 0,
            }
        )

    return samples


async def fetch_resolved_contracts(days_back: int = 180, limit: int = 500) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    fetcher = PolymarketFetcher()
    markets: list[dict[str, Any]] = []
    cursor: str | None = None
    while len(markets) < limit:
        params: dict[str, Any] = {"closed": "true", "limit": min(500, limit - len(markets))}
        if cursor:
            params["next_cursor"] = cursor
        payload = await fetcher._get("/markets", params=params)
        page = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(page, list):
            break
        markets.extend(market for market in page if isinstance(market, dict))
        next_cursor = (
            payload.get("next_cursor") or payload.get("nextCursor")
            if isinstance(payload, dict)
            else None
        )
        if not next_cursor or next_cursor == "LTE=":
            break
        cursor = str(next_cursor)

    resolved: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        condition_id = _market_condition_id(market)
        if not condition_id:
            continue
        end_date = _parse_datetime(
            market.get("end_date") or market.get("endDate") or market.get("endDateIso")
        )
        if end_date and end_date < cutoff:
            continue
        prices: list[float] = []
        timestamps: list[int] = []
        for history_id in _market_history_ids(market):
            prices, timestamps = await fetcher.fetch_price_history(history_id)
            if len(prices) >= settings.min_price_history_hours:
                break
        prices, timestamps = resample_price_history(prices, timestamps)
        if len(prices) < settings.min_price_history_hours:
            continue
        resolved.append(
            {
                "condition_id": condition_id,
                "question": market.get("question") or market.get("title") or "Unknown contract",
                "category": _market_category(market),
                "resolved_value": float(market.get("resolved_value") or market.get("resolvedValue") or 0),
                "prices": prices,
                "timestamps": timestamps,
                "volume_24h": _market_volume_24h(market),
                "duration_days": max(len(prices) / 24, 0),
            }
        )
    return resolved


def _prepare_dataframe(samples: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(samples)
    if df.empty:
        raise ValueError("No training samples generated")
    df = pd.get_dummies(df, columns=["category"])
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0
    return df


def train_signal_classifier(samples: list[dict[str, Any]], output_path: Path | None = None) -> RandomForestClassifier:
    df = _prepare_dataframe(samples)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(
        ascending=False
    )
    print("\nFeature Importance:\n", importance)

    output_path = output_path or settings.model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


async def build_samples(days_back: int, limit: int) -> list[dict[str, Any]]:
    contracts = await fetch_resolved_contracts(days_back=days_back, limit=limit)
    samples: list[dict[str, Any]] = []
    for contract in contracts:
        samples.extend(generate_training_samples(contract))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FFT signal classifier")
    parser.add_argument("--days-back", type=int, default=180)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path, default=settings.model_path)
    args = parser.parse_args()

    samples = asyncio.run(build_samples(args.days_back, args.limit))
    train_signal_classifier(samples, args.output)


if __name__ == "__main__":
    main()
