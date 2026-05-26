from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from training import fetch_resolved_contracts, generate_training_samples


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    hits = sum(1 for sample in samples if int(sample["label"]) == 1)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"signals": 0, "hits": 0})

    for sample in samples:
        category = str(sample.get("category") or "uncategorized")
        by_category[category]["signals"] += 1
        by_category[category]["hits"] += int(sample["label"])

    return {
        "signals": total,
        "hits": hits,
        "hit_rate": _rate(hits, total),
        "false_positive_rate": 1 - hits / total if total else None,
        "minimum_prd_hit_rate": 0.60,
        "passes_prd_threshold": _rate(hits, total) is not None and _rate(hits, total) >= 0.60,
        "by_category": {
            category: {
                **values,
                "hit_rate": _rate(values["hits"], values["signals"]),
            }
            for category, values in sorted(by_category.items())
        },
    }


async def run_backtest(days_back: int, limit: int) -> dict[str, Any]:
    contracts = await fetch_resolved_contracts(days_back=days_back, limit=limit)
    samples: list[dict[str, Any]] = []
    for contract in contracts:
        samples.extend(generate_training_samples(contract))
    summary = summarize_samples(samples)
    summary["contracts"] = len(contracts)
    summary["days_back"] = days_back
    summary["market_limit"] = limit
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest FFT anomaly reversion signals")
    parser.add_argument("--days-back", type=int, default=180)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = asyncio.run(run_backtest(args.days_back, args.limit))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

