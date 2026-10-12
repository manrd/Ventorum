# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Stdio MCP client for the Ventorum server (JSON-RPC 2.0, one message per line).

The client starts ``python -m ventorum.agent --mcp`` as a subprocess, reads
its output with a reader thread (a pipe read with a timeout does not work on
Windows without one) and sends ``initialize``, ``notifications/initialized``,
``tools/list`` and ``tools/call`` requests.

The server environment follows the isolation rules of the benchmark:

* ``VENTORUM_AGENT_AUDIT_LOG`` is a file in the run folder (an independent
  record of every call);
* ``VENTORUM_CONFIG_DIR`` is a temporary copy of the tuning profile folder, so
  that a call to ``ventorum_tune_machine`` cannot change the profile of the
  owner;
* the working folder is an empty temporary folder;
* environment variables whose names look like secrets are removed, and so
  are the key variables that the config names (``drop``).
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "ventorum-agent-benchmark", "version": "1.0"}
SECRET_ENV_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)


class McpError(RuntimeError):
    """A JSON-RPC error reply. ``code`` is the JSON-RPC error code."""

    def __init__(self, code: int, message: str):
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message


class McpTimeout(TimeoutError):
    """The server did not reply in time."""


class McpServerExit(RuntimeError):
    """The server process stopped."""


def repo_root_of_ventorum() -> Path:
    """Return the folder that holds the importable ``ventorum`` package."""
    import importlib.util

    spec = importlib.util.find_spec("ventorum")
    if spec is None or not spec.origin:
        raise RuntimeError("the ventorum package is not importable")
    return Path(spec.origin).resolve().parents[1]


def make_server_env(audit_log: str | Path | None = None, config_dir: str | Path | None = None,
                    extra: dict[str, str] | None = None, drop: Any = ()) -> dict[str, str]:
    """Return the environment of an isolated server process.

    Parameters
    ----------
    audit_log : str or Path, optional
        File for ``VENTORUM_AGENT_AUDIT_LOG``.
    config_dir : str or Path, optional
        Folder for ``VENTORUM_CONFIG_DIR`` (a sandbox copy of the profile).
    extra : dict, optional
        More variables (for example ``MPLBACKEND``).
    drop : iterable of str, optional
        Names of more variables to remove, whatever their names are (for
        example the ``api_key_env`` names of the config).

    Returns
    -------
    dict
        The environment. Variables whose names look like secrets, and the
        variables in *drop*, are not in it.
    """
    dropped = set(drop or ())
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV_NAME.search(k) and k not in dropped}
    root = str(repo_root_of_ventorum())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("VENTORUM_AGENT_AUDIT_LOG", None)
    if audit_log is not None:
        env["VENTORUM_AGENT_AUDIT_LOG"] = str(audit_log)
    if config_dir is not None:
        env["VENTORUM_CONFIG_DIR"] = str(config_dir)
    if extra:
        env.update(extra)
    return env


def sandbox_config_dir(dest: str | Path) -> Path:
    """Copy the current tuning profile folder to *dest* and return *dest*.

    If no profile exists, *dest* is an empty folder. The source folder is
    the one that :func:`ventorum.hardware.profile.config_dir` gives in this
    process.
    """
    from ventorum.hardware.profile import config_dir

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = config_dir()
    if src.is_dir() and src.resolve() != dest.resolve():
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
    return dest


class McpClient:
    """A stdio MCP client that owns one server process.

    Parameters
    ----------
    command : list of str, optional
        Server command. The default is ``[sys.executable, "-m",
        "ventorum.agent", "--mcp"]``.
    env : dict, optional
        Server environment. The default is :func:`make_server_env` with no
        audit log.
    cwd : str or Path, optional
        Working folder of the server. The default is a new empty temporary
        folder, removed at :meth:`close`.
    timeout_s : float, optional
        Default timeout of one request [s].
    """

    def __init__(self, command: list[str] | None = None, env: dict[str, str] | None = None,
                 cwd: str | Path | None = None, timeout_s: float = 300.0):
        self.command = command or [sys.executable, "-m", "ventorum.agent", "--mcp"]
        self.env = env if env is not None else make_server_env()
        self._own_cwd = cwd is None
        self.cwd = Path(cwd) if cwd is not None else Path(tempfile.mkdtemp(prefix="vt_mcp_cwd_"))
        self.timeout_s = float(timeout_s)
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._next_id = 1
        self._stderr_tail: list[str] = []
        self.server_info: dict[str, Any] = {}
        self.protocol_version: str | None = None
        self.tools: list[dict[str, Any]] = []
        self.tools_list_bytes = 0

    # -- process -------------------------------------------------------------
    def start(self) -> McpClient:
        """Start the server process and the reader threads."""
        self.proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, cwd=str(self.cwd), bufsize=-1,  # buffered pipes; _send flushes stdin
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        return self

    def _read_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            self._lines.put(raw)
        self._lines.put(None)

    def _read_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for raw in iter(self.proc.stderr.readline, b""):
            self._stderr_tail.append(raw.decode("utf-8", errors="replace").rstrip())
            del self._stderr_tail[:-50]

    @property
    def alive(self) -> bool:
        """True while the server process runs."""
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> None:
        """Stop the server process at once."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        """Close the input of the server, wait for it to stop, and clean up."""
        if self.proc is not None:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.kill()
        if self._own_cwd:
            shutil.rmtree(self.cwd, ignore_errors=True)

    def __enter__(self) -> McpClient:
        if self.proc is None:
            self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- JSON-RPC --------------------------------------------------------------
    def _send(self, msg: dict[str, Any]) -> None:
        if not self.alive:
            raise McpServerExit("the MCP server is not running. " + " | ".join(self._stderr_tail[-5:]))
        assert self.proc is not None and self.proc.stdin is not None
        data = (json.dumps(msg, allow_nan=False) + "\n").encode("utf-8")
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except OSError as exc:
            raise McpServerExit(f"cannot write to the MCP server: {exc}") from exc

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification (no reply)."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def request(self, method: str, params: dict[str, Any] | None = None,
                timeout_s: float | None = None) -> tuple[Any, int]:
        """Send a request and wait for its reply.

        Parameters
        ----------
        method : str
            JSON-RPC method.
        params : dict, optional
            Parameters.
        timeout_s : float, optional
            Timeout [s]. The default is the client timeout.

        Returns
        -------
        tuple
            ``(result, n_bytes)``: the ``result`` member of the reply and the
            size of the reply line in bytes.

        Raises
        ------
        McpError
            The reply is a JSON-RPC error.
        McpTimeout
            No reply in time.
        McpServerExit
            The server stopped.
        """
        req_id = self._next_id
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else float(timeout_s))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpTimeout(f"no reply to {method} in time")
            try:
                raw = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if not self.alive and self._lines.empty():
                    raise McpServerExit("the MCP server stopped. " + " | ".join(self._stderr_tail[-5:])) from None
                continue
            if raw is None:
                raise McpServerExit("the MCP server closed its output. " + " | ".join(self._stderr_tail[-5:]))
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue  # not a protocol line
            if not isinstance(reply, dict) or reply.get("id") != req_id:
                continue
            if "error" in reply:
                err = reply["error"] or {}
                raise McpError(int(err.get("code", 0)), str(err.get("message", "")))
            return reply.get("result"), len(raw)

    # -- MCP -------------------------------------------------------------------
    def initialize(self, protocol_version: str = PROTOCOL_VERSION, timeout_s: float = 120.0) -> dict[str, Any]:
        """Send ``initialize`` and ``notifications/initialized``. Return the result."""
        result, _ = self.request("initialize", {"protocolVersion": protocol_version, "capabilities": {},
                                                "clientInfo": CLIENT_INFO}, timeout_s=timeout_s)
        self.server_info = result.get("serverInfo", {})
        self.protocol_version = result.get("protocolVersion")
        self.notify("notifications/initialized")
        return result

    def list_tools(self, timeout_s: float = 120.0) -> list[dict[str, Any]]:
        """Send ``tools/list`` and return the tools (name, description, inputSchema)."""
        result, n = self.request("tools/list", {}, timeout_s=timeout_s)
        self.tools = list(result.get("tools", []))
        self.tools_list_bytes = n
        return self.tools

    def call_tool(self, name: str, arguments: Any, timeout_s: float | None = None) -> dict[str, Any]:
        """Send ``tools/call`` and return the outcome.

        Parameters
        ----------
        name : str
            Tool name.
        arguments : Any
            Tool arguments (normally a dict).
        timeout_s : float, optional
            Timeout [s].

        Returns
        -------
        dict
            ``{"is_error", "text", "payload", "rpc_error", "bytes"}``.
            ``payload`` is the parsed tool payload (None if the text is not
            JSON). ``rpc_error`` is ``{"code", "message"}`` for a JSON-RPC
            error (for example an unknown tool), else None.

        Raises
        ------
        McpTimeout, McpServerExit
            As :meth:`request`.
        """
        try:
            result, n = self.request("tools/call", {"name": name, "arguments": arguments}, timeout_s=timeout_s)
        except McpError as exc:
            return {"is_error": True, "text": exc.message, "payload": None,
                    "rpc_error": {"code": exc.code, "message": exc.message}, "bytes": 0}
        text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        return {"is_error": bool(result.get("isError")), "text": text, "payload": payload,
                "rpc_error": None, "bytes": n}


def start_session(audit_log: str | Path | None = None, config_dir: str | Path | None = None,
                  timeout_s: float = 300.0, drop_env: Any = ()) -> McpClient:
    """Start an isolated server, initialize it and list its tools.

    Parameters
    ----------
    audit_log : str or Path, optional
        Audit log file of the server.
    config_dir : str or Path, optional
        Sandbox profile folder.
    timeout_s : float, optional
        Default request timeout [s].
    drop_env : iterable of str, optional
        Names of environment variables that the server must not get (the
        key variables of the config).

    Returns
    -------
    McpClient
        A client that is ready for ``tools/call``.
    """
    client = McpClient(env=make_server_env(audit_log, config_dir, drop=drop_env), timeout_s=timeout_s).start()
    try:
        client.initialize()
        client.list_tools()
    except Exception:
        client.kill()
        client.close()
        raise
    return client


def other_mcp_servers() -> int | None:
    """Count other running ``ventorum.agent --mcp`` processes (best effort).

    Returns
    -------
    int or None
        The number of such processes, or None if the process list is not
        available.
    """
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"]
        else:
            cmd = ["ps", "-eo", "args"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(1 for line in out.splitlines() if "ventorum.agent" in line and "--mcp" in line)
