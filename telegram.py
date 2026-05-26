from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from urllib.request import Request, urlopen

from config import Settings, settings
from db import is_suppressed, mark_alert_sent, mark_suppressed, recently_alerted


def _escape_markdown_v2(value: str) -> str:
    reserved = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{char}" if char in reserved else char for char in value)


def _market_url(contract: dict[str, Any]) -> str:
    raw = contract.get("market_url")
    if isinstance(raw, str) and raw.startswith("http"):
        return raw
    if isinstance(raw, str) and raw:
        path = raw.strip()
        if path.startswith("/"):
            return f"https://polymarket.com{path}"
        if path.startswith("event/") or path.startswith("market/"):
            return f"https://polymarket.com/{path}"
        return f"https://polymarket.com/event/{path.strip('/')}"
    return f"https://polymarket.com/event/{contract['condition_id']}"


def build_alert_message(contract: dict[str, Any], features: dict[str, Any]) -> str:
    current_price = float(contract["prices"][-1])
    implied_prob = round(current_price * 100, 1)
    cycle_str = (
        f"{float(features['cycle_days']):.1f} days"
        if features.get("cycle_days")
        else "N/A"
    )
    detected_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    question = _escape_markdown_v2(str(contract["question"]))
    category = _escape_markdown_v2(str(contract["category"]).capitalize())
    market_url = _escape_markdown_v2(_market_url(contract))

    return (
        f"⚡ *FFT ANOMALY DETECTED*\n\n"
        f"📋 *Contract:* {question}\n"
        f"🏷️ *Category:* {category}\n"
        f"💰 *Current Price:* ${current_price:.2f} \\({implied_prob}% implied prob\\)\n\n"
        f"📊 *Signal Data:*\n"
        f"• Deviation Score: `{float(features['deviation_score']):.2f}` \\(threshold: 2\\.5\\)\n"
        f"• Dominant Cycle: `{_escape_markdown_v2(cycle_str)}`\n"
        f"• Noise Ratio: `{float(features['noise_ratio']):.2f}` \\(low = clean signal\\)\n"
        f"• Amplitude: `{float(features['amplitude']):.4f}`\n\n"
        f"📅 *Detected:* {detected_at}\n"
        f"🔗 {market_url}\n\n"
        f"⚠️ _Alert only — no trade executed \\(Phase 1\\)_"
    )


async def send_telegram_alert(
    contract: dict[str, Any],
    features: dict[str, Any],
    cfg: Settings = settings,
) -> bool:
    if cfg.disable_telegram:
        return False
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required to send alerts")

    message = build_alert_message(contract, features)
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    payload = json.dumps(
        {
                "chat_id": cfg.telegram_chat_id,
                "text": message,
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True,
        }
    ).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    response = await __import__("asyncio").to_thread(
        urlopen, request, timeout=cfg.request_timeout_seconds
    )
    return int(response.status) == 200


async def maybe_send_alert(
    contract: dict[str, Any],
    features: dict[str, Any],
    cfg: Settings = settings,
) -> bool:
    condition_id = str(contract["condition_id"])
    if is_suppressed(condition_id, cfg.suppression_window_hours, cfg.suppression_alert_count):
        mark_suppressed(condition_id)
        return False
    if recently_alerted(condition_id, cfg.alert_cooldown_hours):
        return False

    sent = await send_telegram_alert(contract, features, cfg)
    if sent:
        mark_alert_sent(condition_id)
    return sent
