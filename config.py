from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows DB/schema commands to run before dependencies are installed.
    def load_dotenv(path: object = None, *_args: object, **_kwargs: object) -> bool:
        if path is None:
            return False
        env_path = Path(path)
        if not env_path.exists():
            return False
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return True

        return False


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    polymarket_clob_url: str = os.getenv(
        "POLYMARKET_CLOB_URL", "https://clob.polymarket.com"
    ).rstrip("/")
    telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = os.getenv("TELEGRAM_CHAT_ID")

    run_interval_hours: int = int(os.getenv("RUN_INTERVAL_HOURS", "6"))
    run_interval_minutes: int = int(
        os.getenv("RUN_INTERVAL_MINUTES", str(int(os.getenv("RUN_INTERVAL_HOURS", "6")) * 60))
    )
    min_volume_24h: float = float(os.getenv("MIN_VOLUME_24H", "10000"))
    min_price_history_hours: int = int(os.getenv("MIN_PRICE_HISTORY_HOURS", "336"))
    resolution_buffer_hours: int = int(os.getenv("RESOLUTION_BUFFER_HOURS", "48"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
    request_concurrency: int = int(os.getenv("REQUEST_CONCURRENCY", "6"))
    request_spacing_seconds: float = float(os.getenv("REQUEST_SPACING_SECONDS", "0.15"))
    max_markets: int | None = (
        int(os.getenv("MAX_MARKETS")) if os.getenv("MAX_MARKETS") else None
    )

    anomaly_deviation_threshold: float = float(
        os.getenv("ANOMALY_DEVIATION_THRESHOLD", "2.5")
    )
    anomaly_noise_max: float = float(os.getenv("ANOMALY_NOISE_MAX", "0.65"))
    anomaly_amplitude_min: float = float(os.getenv("ANOMALY_AMPLITUDE_MIN", "0.02"))
    efficient_amplitude_min: float = float(os.getenv("EFFICIENT_AMPLITUDE_MIN", "0.01"))
    noisy_noise_threshold: float = float(os.getenv("NOISY_NOISE_THRESHOLD", "0.80"))
    trending_cycle_days_min: float = float(os.getenv("TRENDING_CYCLE_DAYS_MIN", "5"))
    trending_noise_max: float = float(os.getenv("TRENDING_NOISE_MAX", "0.40"))
    trending_amplitude_min: float = float(os.getenv("TRENDING_AMPLITUDE_MIN", "0.03"))

    alert_cooldown_hours: int = int(os.getenv("ALERT_COOLDOWN_HOURS", "24"))
    suppression_window_hours: int = int(os.getenv("SUPPRESSION_WINDOW_HOURS", "72"))
    suppression_alert_count: int = int(os.getenv("SUPPRESSION_ALERT_COUNT", "3"))
    disable_telegram: bool = _get_bool("DISABLE_TELEGRAM", False)

    database_path: Path = Path(os.getenv("DATABASE_PATH", BASE_DIR / "signals.db"))
    model_path: Path = Path(
        os.getenv("MODEL_PATH", BASE_DIR / "models" / "signal_classifier_current.pkl")
    )
    classifier_confidence_threshold: float = float(
        os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.65")
    )
    watch_confidence_threshold: float = float(
        os.getenv("WATCH_CONFIDENCE_THRESHOLD", "0.45")
    )
    health_check_enabled: bool = _get_bool("HEALTH_CHECK_ENABLED", True)
    port: int = int(os.getenv("PORT", "8080"))


settings = Settings()
