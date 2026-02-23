from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from qimg.mcp.protocol import MCPProtocol
from qimg.service import QimgService


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qimg-mcp"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, {"ok": True})
            return
        self._write_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/mcp":
            self._write_json(404, {"ok": False, "error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json(400, {"ok": False, "error": "invalid json"})
            return

        protocol: MCPProtocol = self.server.protocol  # type: ignore[attr-defined]
        response = protocol.handle(payload)
        self._write_json(200, response)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep CLI output clean.
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_http_server(service: QimgService, host: str = "127.0.0.1", port: int = 8181) -> int:
    protocol = MCPProtocol(service)

    class _Srv(ThreadingHTTPServer):
        protocol = protocol

    server = _Srv((host, port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
