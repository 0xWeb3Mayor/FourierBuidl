from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from config import settings
from db import get_latest_signals, get_signal_summary, init_db


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/health"}:
            self._send_json({"status": "ok", "service": "fft-signal-agent"})
            return

        init_db()
        if parsed.path == "/signals/latest":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["50"])[0])
            self._send_json({"signals": get_latest_signals(limit=limit)})
            return

        if parsed.path == "/signals/anomalies":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["50"])[0])
            self._send_json(
                {"signals": get_latest_signals(limit=limit, signal_class="ANOMALY")}
            )
            return

        if parsed.path == "/signals/summary":
            self._send_json(get_signal_summary())
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, payload: dict[str, object]) -> None:
        try:
            init_db()
        except Exception:
            if "status" not in payload:
                self.send_response(500)
                self.end_headers()
                return
        body = json.dumps(payload, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def start_health_server(port: int = settings.port) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
