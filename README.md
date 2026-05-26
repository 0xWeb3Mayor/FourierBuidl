# FFT Signal Agent

Phase 1 implementation of the FFT Signal Agent from `PRD_01_FFT_Signal_Agent (1).md`.

It monitors active Polymarket contracts, pulls hourly price histories, extracts FFT features, classifies each contract with the PRD rule engine, logs every evaluated contract to SQLite, and sends Telegram alerts for anomaly signals with cooldown and suppression.

## Setup

```bash
cd fft-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISABLE_TELEGRAM=false
```

## Run

Initialize the database:

```bash
python main.py init-db
```

Run a single scan:

```bash
python main.py run-once
```

Run continuously on the configured interval:

```bash
python main.py scheduler
```

When `HEALTH_CHECK_ENABLED=true`, the scheduler also exposes `GET /health` on `PORT` for Railway or UptimeRobot.

`RUN_INTERVAL_MINUTES` controls scan cadence and overrides the older `RUN_INTERVAL_HOURS` setting. The local default is 60 minutes.

## Live Analysis API

When the health server is enabled, these JSON endpoints are available:

```bash
GET /signals/latest?limit=50
GET /signals/anomalies?limit=50
GET /signals/summary
```

Use these from your frontend after deploying the backend to Render.

## Database

The SQLite schema matches the PRD `signal_log` table. Every evaluated contract is logged, not just anomalies. Alerts are tracked in `alert_log` to enforce:

- 24 hour cooldown per contract
- suppression after 3 alerts in 72 hours, retained until the contract is resolved or manually cleared

## Training And Calibration

Section 16 is implemented in `training.py` and `ml.py`.

Build samples and train the RandomForest classifier:

```bash
python training.py --days-back 180 --limit 500
```

The model is saved to `models/signal_classifier_current.pkl`. Live Phase 1 still uses the rule-based classifier by default, matching the PRD. `ml.py` provides the Phase 2-ready classifier wrapper.

Run a historical anomaly reversion backtest:

```bash
python backtest.py --days-back 180 --limit 500
```

The PRD requires at least a 60% 72-hour anomaly reversion hit rate before treating the signal as Phase 2-ready.

## Render

Use this start command:

```bash
python main.py scheduler
```

Set environment variables in Railway rather than committing secrets.
