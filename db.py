from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_timestamp INTEGER,
  condition_id TEXT,
  question TEXT,
  category TEXT,
  current_price REAL,
  dominant_freq REAL,
  cycle_days REAL,
  amplitude REAL,
  noise_ratio REAL,
  deviation_score REAL,
  signal_class TEXT,
  alert_sent INTEGER DEFAULT 0,
  resolved INTEGER DEFAULT 0,
  resolved_price REAL
);

CREATE INDEX IF NOT EXISTS idx_signal_log_run_timestamp
ON signal_log(run_timestamp);

CREATE INDEX IF NOT EXISTS idx_signal_log_condition_id
ON signal_log(condition_id);

CREATE TABLE IF NOT EXISTS alert_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  condition_id TEXT NOT NULL,
  alert_timestamp INTEGER NOT NULL,
  suppressed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alert_log_condition_ts
ON alert_log(condition_id, alert_timestamp);
"""


def connect(db_path: Path = settings.database_path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = settings.database_path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def log_signal(
    contract: dict[str, Any],
    features: dict[str, Any],
    signal_class: str,
    alert_sent: bool = False,
    run_timestamp: int | None = None,
    db_path: Path = settings.database_path,
) -> int:
    run_timestamp = run_timestamp or int(time.time() * 1000)
    prices = contract.get("prices") or []
    current_price = float(prices[-1]) if prices else None

    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO signal_log (
              run_timestamp, condition_id, question, category, current_price,
              dominant_freq, cycle_days, amplitude, noise_ratio, deviation_score,
              signal_class, alert_sent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_timestamp,
                contract.get("condition_id"),
                contract.get("question"),
                contract.get("category"),
                current_price,
                features.get("dominant_freq"),
                features.get("cycle_days"),
                features.get("amplitude"),
                features.get("noise_ratio"),
                features.get("deviation_score"),
                signal_class,
                1 if alert_sent else 0,
            ),
        )
        return int(cursor.lastrowid)


def update_signal_alert_sent(row_id: int, db_path: Path = settings.database_path) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE signal_log SET alert_sent = 1 WHERE id = ?", (row_id,))


def mark_alert_sent(condition_id: str, db_path: Path = settings.database_path) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO alert_log (condition_id, alert_timestamp, suppressed) VALUES (?, ?, 0)",
            (condition_id, int(time.time() * 1000)),
        )


def recently_alerted(
    condition_id: str,
    cooldown_hours: int = settings.alert_cooldown_hours,
    db_path: Path = settings.database_path,
) -> bool:
    cutoff = int((time.time() - cooldown_hours * 3600) * 1000)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM alert_log
            WHERE condition_id = ? AND alert_timestamp >= ? AND suppressed = 0
            LIMIT 1
            """,
            (condition_id, cutoff),
        ).fetchone()
    return row is not None


def is_suppressed(
    condition_id: str,
    window_hours: int = settings.suppression_window_hours,
    alert_count: int = settings.suppression_alert_count,
    db_path: Path = settings.database_path,
) -> bool:
    cutoff = int((time.time() - window_hours * 3600) * 1000)
    with connect(db_path) as conn:
        suppressed_row = conn.execute(
            """
            SELECT 1 FROM alert_log
            WHERE condition_id = ? AND suppressed = 1
            LIMIT 1
            """,
            (condition_id,),
        ).fetchone()
        if suppressed_row is not None:
            return True

        row = conn.execute(
            """
            SELECT COUNT(*) AS count FROM alert_log
            WHERE condition_id = ? AND alert_timestamp >= ?
            """,
            (condition_id, cutoff),
        ).fetchone()
    return int(row["count"]) >= alert_count if row else False


def mark_suppressed(condition_id: str, db_path: Path = settings.database_path) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO alert_log (condition_id, alert_timestamp, suppressed) VALUES (?, ?, 1)",
            (condition_id, int(time.time() * 1000)),
        )
