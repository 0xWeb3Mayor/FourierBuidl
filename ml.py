from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from classifier import classify_signal
from config import Settings, settings


FEATURE_COLUMNS = [
    "deviation_score",
    "noise_ratio",
    "amplitude",
    "cycle_days",
    "current_price",
    "volume",
    "duration_days",
    "category_crypto",
    "category_sports",
    "category_science",
    "category_politics",
]

TARGET_COLUMN = "label"


def build_feature_row(features: dict[str, Any], contract: dict[str, Any]) -> list[float]:
    category = str(contract.get("category") or "").lower()
    prices = contract.get("prices") or []
    return [
        float(features.get("deviation_score") or 0),
        float(features.get("noise_ratio") or 1),
        float(features.get("amplitude") or 0),
        float(features.get("cycle_days") or 0),
        float(prices[-1]) if prices else float(contract.get("current_price") or 0),
        float(contract.get("volume") or contract.get("volume_24h") or 0),
        float(contract.get("duration_days") or 0),
        1.0 if category == "crypto" else 0.0,
        1.0 if category == "sports" else 0.0,
        1.0 if category == "science" else 0.0,
        1.0 if category == "politics" else 0.0,
    ]


def load_classifier(model_path: Path = settings.model_path) -> Any | None:
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def classify_signal_ml(
    features: dict[str, Any],
    contract: dict[str, Any],
    model: Any | None = None,
    cfg: Settings = settings,
) -> tuple[str, float | None]:
    model = model if model is not None else load_classifier(cfg.model_path)
    if model is None:
        return classify_signal(features, cfg), None

    row = build_feature_row(features, contract)
    confidence = float(model.predict_proba([row])[0][1])
    if confidence >= cfg.classifier_confidence_threshold:
        return "ANOMALY", confidence
    if confidence >= cfg.watch_confidence_threshold:
        return "WATCH", confidence
    return "EFFICIENT", confidence

