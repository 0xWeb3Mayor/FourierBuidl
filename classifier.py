from __future__ import annotations

from typing import Any

from config import Settings, settings


SIGNAL_CLASSES = {"ANOMALY", "TRENDING", "NOISY", "EFFICIENT", "WATCH"}


def classify_signal(features: dict[str, Any], cfg: Settings = settings) -> str:
    """
    Rule-based Phase 1 classifier from the PRD.
    """
    dev = float(features.get("deviation_score") or 0)
    noise = float(features.get("noise_ratio") or 1)
    amp = float(features.get("amplitude") or 0)
    cycle = features.get("cycle_days")
    cycle_days = float(cycle) if cycle is not None else None

    if amp < cfg.efficient_amplitude_min:
        return "EFFICIENT"

    if noise > cfg.noisy_noise_threshold:
        return "NOISY"

    if (
        dev > cfg.anomaly_deviation_threshold
        and noise < cfg.anomaly_noise_max
        and amp > cfg.anomaly_amplitude_min
    ):
        return "ANOMALY"

    if (
        cycle_days
        and cycle_days > cfg.trending_cycle_days_min
        and noise < cfg.trending_noise_max
        and amp > cfg.trending_amplitude_min
    ):
        return "TRENDING"

    return "EFFICIENT"

