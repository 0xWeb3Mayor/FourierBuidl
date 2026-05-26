from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
from typing import Any

from config import settings
from db import (
    get_outcome_candidates,
    init_db,
    log_signal,
    update_signal_alert_sent,
    update_signal_outcome,
)
from outcomes import evaluate_signal_outcome


def update_post_alert_outcomes(contract: dict[str, Any]) -> None:
    timestamps = contract.get("timestamps") or []
    prices = contract.get("prices") or []
    if not timestamps or not prices:
        return
    for signal in get_outcome_candidates(str(contract["condition_id"])):
        outcome = evaluate_signal_outcome(signal, prices, timestamps)
        if outcome:
            update_signal_outcome(int(signal["id"]), outcome)


async def run_agent_once() -> list[dict[str, Any]]:
    from classifier import classify_signal
    from fetcher import fetch_active_contracts
    from fft_engine import estimate_reversion_target, run_fft_analysis
    from telegram import maybe_send_alert

    init_db()
    contracts = await fetch_active_contracts()
    print(f"[{datetime.now()}] fetched {len(contracts)} contracts eligible for FFT analysis")
    results: list[dict[str, Any]] = []
    run_timestamp = int(time.time() * 1000)

    for contract in contracts:
        if len(contract["prices"]) < settings.min_price_history_hours:
            continue

        update_post_alert_outcomes(contract)

        try:
            features = run_fft_analysis(contract["prices"])
            target_price, direction = estimate_reversion_target(contract["prices"], features)
            features = {
                **features,
                "target_price": target_price,
                "direction": direction,
            }
            signal_class = classify_signal(features)
        except Exception as exc:
            print(f"[{datetime.now()}] skipped {contract.get('condition_id')}: {exc}")
            continue

        row_id = log_signal(contract, features, signal_class, False, run_timestamp)
        alert_sent = False

        if signal_class == "ANOMALY":
            try:
                alert_sent = await maybe_send_alert(contract, features)
            except RuntimeError as exc:
                print(f"[{datetime.now()}] alert not sent: {exc}")
            except Exception as exc:
                print(f"[{datetime.now()}] alert failed for {contract['condition_id']}: {exc}")
            if alert_sent:
                update_signal_alert_sent(row_id)

        results.append({**contract, **features, "signal_class": signal_class, "alert_sent": alert_sent})

    anomaly_count = sum(1 for result in results if result["signal_class"] == "ANOMALY")
    print(
        f"[{datetime.now()}] run complete - {len(results)} contracts analyzed, "
        f"{anomaly_count} anomalies"
    )
    return results


def run_agent() -> None:
    asyncio.run(run_agent_once())


def start_scheduler() -> None:
    init_db()
    if settings.health_check_enabled:
        from health import start_health_server

        start_health_server(settings.port)
        print(f"[{datetime.now()}] health endpoint listening on /health port {settings.port}")

    print(
        f"[{datetime.now()}] FFT Signal Agent starting; "
        f"interval={settings.run_interval_minutes}m"
    )
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        scheduler = BlockingScheduler()
        scheduler.add_job(
            run_agent,
            IntervalTrigger(minutes=settings.run_interval_minutes),
            id="fft_agent",
            replace_existing=True,
            next_run_time=datetime.now(),
        )
        scheduler.start()
    except ImportError:
        while True:
            run_agent()
            time.sleep(settings.run_interval_minutes * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="FFT Signal Agent with Telegram alerts")
    parser.add_argument(
        "command",
        nargs="?",
        choices={"run-once", "scheduler", "init-db"},
        default="scheduler",
        help="Run one scan, start scheduler, or initialize SQLite schema",
    )
    args = parser.parse_args()

    if args.command == "run-once":
        run_agent()
    elif args.command == "init-db":
        init_db()
        print(f"Initialized database at {settings.database_path}")
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
