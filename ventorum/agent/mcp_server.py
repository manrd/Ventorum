# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Model Context Protocol (MCP) server over stdio (JSON-RPC 2.0, one message per line).

Protocol version 2025-06-18 (2024-11-05 and 2025-03-26 are also accepted).
On ``initialize`` the server replies with the version that the client
requests if it supports it, else with the latest version it supports.

Methods: ``initialize``, ``ping``, ``tools/list``, ``tools/call``.
Notifications (messages without ``id``) get no reply. Every request with an
``id`` gets exactly one reply. Error codes:

* -32700 parse error (invalid JSON, also NaN and Infinity; also bytes that are
  not valid UTF-8; the reply has ``id: null``)
* -32600 invalid request (not an object, a JSON array, no ``method``, an ``id``
  that is null or not a string or an integer)
* -32601 method not found
* -32602 invalid params (``params`` not an object, bad tool call params, unknown tool)
* -32603 internal error

JSON-RPC batches (JSON arrays) are not supported (MCP 2025-06-18 removed
them); an array gets -32600.

A tool that runs and fails (bad input, ground strike, refused method) gives
a normal ``tools/call`` result with ``isError: true`` and the error payload
in ``content``.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from ventorum.agent.dispatcher import call_tool, resolve_tool_name
from ventorum.agent.schemas import get_tool_schemas

SERVER_NAME = "ventorum-aerodynamics"
SERVER_VERSION = "0.3.0"
LATEST_PROTOCOL = "2025-06-18"
SUPPORTED_PROTOCOLS = ("2024-11-05", "2025-03-26", LATEST_PROTOCOL)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class _RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _dispatch(method: str, params: dict[str, Any]) -> Any:
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else LATEST_PROTOCOL
        return {
            "protocolVersion": version,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"tools": {}},
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": get_tool_schemas(format="mcp")}
    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            raise _RpcError(INVALID_PARAMS, "tools/call needs 'params.name' (a string).")
        if resolve_tool_name(name) is None:
            raise _RpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        args = params.get("arguments", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise _RpcError(INVALID_PARAMS, "'params.arguments' must be an object.")
        payload = call_tool(name, args)
        return {
            "content": [{"type": "text", "text": json.dumps(payload, allow_nan=False)}],
            "isError": payload.get("status") != "success",
        }
    raise _RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")


def handle_request(msg: Any) -> dict[str, Any] | None:
    """Process one decoded JSON-RPC message. Return the reply, or None for a notification."""
    if not isinstance(msg, dict):
        return _error(None, INVALID_REQUEST, "Invalid request: a request must be a JSON object.")
    is_notification = "id" not in msg
    req_id = msg.get("id")
    if not is_notification and not (isinstance(req_id, (str, int)) and not isinstance(req_id, bool)):
        return _error(None, INVALID_REQUEST, "Invalid request: 'id' must be a string or an integer (not null).")
    method = msg.get("method")
    if msg.get("jsonrpc") != "2.0" or not isinstance(method, str):
        if is_notification:
            return None
        return _error(req_id, INVALID_REQUEST, "Invalid request: needs \"jsonrpc\": \"2.0\" and a string 'method'.")
    params = msg.get("params", {})
    if not isinstance(params, dict):
        if is_notification:
            return None
        return _error(req_id, INVALID_PARAMS, "Invalid params: 'params' must be an object.")
    if is_notification:
        # Notifications (for example notifications/initialized) get no reply.
        return None
    try:
        return _result(req_id, _dispatch(method, params))
    except _RpcError as exc:
        return _error(req_id, exc.code, exc.message)
    except Exception as exc:  # noqa: BLE001 - every request gets a reply
        return _error(req_id, INTERNAL_ERROR, f"Internal error: {type(exc).__name__}: {exc}")


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def handle_message(line: str) -> str | None:
    """Process one line of input. Return the reply line (JSON text), or None if no reply is due."""
    try:
        # Strict JSON: NaN, Infinity and -Infinity are refused.
        msg = json.loads(line, parse_constant=_no_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return json.dumps(_error(None, PARSE_ERROR, f"Parse error: {exc}"))
    try:
        if isinstance(msg, list):
            return json.dumps(_error(None, INVALID_REQUEST,
                                     "Invalid request: JSON-RPC batches (arrays) are not supported."))
        reply = handle_request(msg)
        return json.dumps(reply, allow_nan=False) if reply is not None else None
    except Exception as exc:  # noqa: BLE001
        req_id = msg.get("id") if isinstance(msg, dict) else None
        return json.dumps(_error(req_id, INTERNAL_ERROR, f"Internal error: {type(exc).__name__}: {exc}"))


def run_mcp_server(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Run the MCP server on stdin and stdout until the input ends.

    The input is read as bytes and decoded as UTF-8 per line. A line
    that is not valid UTF-8 gets a JSON-RPC parse error (code -32700)
    and the server keeps running.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stream = stdin.buffer if hasattr(stdin, "buffer") else stdin
    for raw in stream:
        if isinstance(raw, bytes):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                reply = json.dumps(_error(None, PARSE_ERROR, f"Parse error: {exc}"))
                stdout.write(reply + "\n")
                stdout.flush()
                continue
        else:
            line = raw
        if not line.strip():
            continue
        reply = handle_message(line)
        if reply is not None:
            stdout.write(reply + "\n")
            stdout.flush()


if __name__ == "__main__":
    run_mcp_server()
