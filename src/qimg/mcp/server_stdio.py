from __future__ import annotations

import json
import sys

from qimg.mcp.protocol import MCPProtocol
from qimg.service import QimgService


def run_stdio_server(service: QimgService) -> int:
    protocol = MCPProtocol(service)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line in {"quit", "exit", "shutdown"}:
            return 0
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "invalid json"}), flush=True)
            continue
        response = protocol.handle(payload)
        print(json.dumps(response), flush=True)

    return 0
