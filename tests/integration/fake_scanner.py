"""Tool-guard scanner stand-in that runs inside the LiteLLM container and records what it was asked."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARKER = "IGNORE PREVIOUS INSTRUCTIONS"
FAIL = "SCANNER-FAILS-ON-THIS"
MAX_BODY_BYTES = int(os.environ.get("FAKE_SCANNER_MAX_BODY_BYTES", "0"))
received = []
verdicts = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path != "/_received":
            self._json(404, {})
            return
        self._json(200, received)

    def do_POST(self):
        if self.path != "/v1/scan":
            self._json(404, {})
            return
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        if MAX_BODY_BYTES and len(raw) > MAX_BODY_BYTES:
            self._json(413, {"error": "request body too large"})
            return
        body = json.loads(raw or b"{}")
        received.append(body)
        if any(FAIL in item["text"] for item in body["new"]):
            self._json(503, {"error": "scanner down"})
            return
        for item in body["new"]:
            verdicts[item["hash"]] = MARKER in item["text"]
        flagged = [i["hash"] for i in body["new"] if verdicts[i["hash"]]]
        flagged += [h for h in body["known"] if verdicts.get(h)]
        unknown = [h for h in body["known"] if h not in verdicts]
        self._json(200, {"flagged": flagged, "unknown": unknown})

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("FAKE_SCANNER_PORT", "8097"))), Handler).serve_forever()
