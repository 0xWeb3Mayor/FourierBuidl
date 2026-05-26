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
  target_price REAL,
  direction TEXT,
  volume_24h REAL,
  spread REAL,
  price_24h REAL,
  price_48h REAL,
  price_72h REAL,
  reverted_24h INTEGER,
  reverted_48h INTEGER,
  reverted_72h INTEGER,
  outcome_checked_at INTEGER,
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


SIGNAL_LOG_MIGRATIONS = {
    "target_price": "ALTER TABLE signal_log ADD COLUMN target_price REAL",
    "direction": "ALTER TABLE signal_log ADD COLUMN direction TEXT",
    "volume_24h": "ALTER TABLE signal_log ADD COLUMN volume_24h REAL",
    "spread": "ALTER TABLE signal_log ADD COLUMN spread REAL",
    "price_24h": "ALTER TABLE signal_log ADD COLUMN price_24h REAL",
    "price_48h": "ALTER TABLE signal_log ADD COLUMN price_48h REAL",
    "price_72h": "ALTER TABLE signal_log ADD COLUMN price_72h REAL",
    "reverted_24h": "ALTER TABLE signal_log ADD COLUMN reverted_24h INTEGER",
    "reverted_48h": "ALTER TABLE signal_log ADD COLUMN reverted_48h INTEGER",
    "reverted_72h": "ALTER TABLE signal_log ADD COLUMN reverted_72h INTEGER",
    "outcome_checked_at": "ALTER TABLE signal_log ADD COLUMN outcome_checked_at INTEGER",
}


def connect(db_path: Path = settings.database_path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = settings.database_path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(signal_log)").fetchall()
        }
        for column, statement in SIGNAL_LOG_MIGRATIONS.items():
            if column not in existing_columns:
                conn.execute(statement)


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
              signal_class, alert_sent, target_price, direction, volume_24h, spread
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_timestamp,
                contract.get("condition_id"),
                contract.get("question"),
                contract.get("category"),
                contract.get("current_price", current_price),
                features.get("dominant_freq"),
                features.get("cycle_days"),
                features.get("amplitude"),
                features.get("noise_ratio"),
                features.get("deviation_score"),
                signal_class,
                1 if alert_sent else 0,
                features.get("target_price"),
                features.get("direction"),
                contract.get("volume_24h"),
                contract.get("spread"),
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


def get_latest_signals(
    limit: int = 50,
    signal_class: str | None = None,
    db_path: Path = settings.database_path,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    where = ""
    params: list[Any] = []
    if signal_class:
        where = "WHERE signal_class = ?"
        params.append(signal_class)
    params.append(limit)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM signal_log
            {where}
            ORDER BY run_timestamp DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_signal_summary(db_path: Path = settings.database_path) -> dict[str, Any]:
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM signal_log").fetchone()
        by_class = conn.execute(
            """
            SELECT signal_class, COUNT(*) AS count
            FROM signal_log
            GROUP BY signal_class
            ORDER BY count DESC
            """
        ).fetchall()
        anomaly_outcomes = conn.execute(
            """
            SELECT
              COUNT(*) AS checked,
              SUM(CASE WHEN reverted_72h = 1 THEN 1 ELSE 0 END) AS hits
            FROM signal_log
            WHERE signal_class = 'ANOMALY' AND reverted_72h IS NOT NULL
            """
        ).fetchone()
    checked = int(anomaly_outcomes["checked"] or 0)
    hits = int(anomaly_outcomes["hits"] or 0)
    return {
        "total_signals": int(total["count"] or 0),
        "by_class": {row["signal_class"]: int(row["count"]) for row in by_class},
        "anomaly_outcomes_72h": {
            "checked": checked,
            "hits": hits,
            "hit_rate": hits / checked if checked else None,
        },
    }


def get_outcome_candidates(
    condition_id: str,
    db_path: Path = settings.database_path,
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM signal_log
            WHERE condition_id = ?
              AND signal_class = 'ANOMALY'
              AND target_price IS NOT NULL
              AND direction IS NOT NULL
              AND (
                reverted_24h IS NULL OR reverted_48h IS NULL OR reverted_72h IS NULL
              )
            ORDER BY run_timestamp ASC
            """,
            (condition_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_signal_outcome(
    row_id: int,
    outcome: dict[str, Any],
    db_path: Path = settings.database_path,
) -> None:
    fields = [
        "price_24h",
        "price_48h",
        "price_72h",
        "reverted_24h",
        "reverted_48h",
        "reverted_72h",
        "outcome_checked_at",
    ]
    assignments = ", ".join(f"{field} = ?" for field in fields if field in outcome)
    values = [outcome[field] for field in fields if field in outcome]
    if not assignments:
        return
    values.append(row_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE signal_log SET {assignments} WHERE id = ?", values)
