"""Serve the map with live model output.

    python -m live.server            # http://localhost:8000
    python -m live.server --retrain  # retrain the models first

Polls Yahoo Finance every --poll seconds, scores with the trained models and
serves the result at /data.json (same shape the static map uses).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from live.engine import ROOT, load_bank, score_all, to_payload

STATE = {"payload": None, "error": None}


def poll_loop(bank: dict, every: int, threshold: float) -> None:
    while True:
        try:
            payload = to_payload(score_all(bank, threshold), bank)
            STATE["payload"] = json.dumps(payload, separators=(",", ":")).encode()
            STATE["error"] = None
            print(f"[{payload['fetched_at'][:19]}] scored, last bar {payload['last_bar']}", flush=True)
        except Exception as e:  # keep serving the last good data
            STATE["error"] = repr(e)
            traceback.print_exc()
        time.sleep(every)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/data.json":
            body = STATE["payload"]
            if body is None:
                self.send_error(503, STATE["error"] or "first fetch in progress")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *a):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--poll", type=int, default=60, help="seconds between feed polls")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    bank = load_bank(args.data, args.horizon, args.retrain)
    threading.Thread(target=poll_loop, args=(bank, args.poll, args.threshold), daemon=True).start()
    handler = partial(Handler, directory=str(ROOT / "frontend"))
    print(f"serving http://localhost:{args.port} (first fetch takes ~10 s)", flush=True)
    ThreadingHTTPServer(("", args.port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
