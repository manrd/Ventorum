# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Model adapters of the agent benchmark (standard library only).

Each adapter sends the conversation and the tool list to a model and returns
one :class:`ModelReply` (text, tool calls, token usage, duration).

* ``openai_chat``: the OpenAI chat-completions format. It covers Ollama
  (``/v1/chat/completions``), OpenAI, the OpenAI-compatible endpoint of
  Google Gemini, and other gateways.
* ``anthropic_messages``: the Anthropic Messages API, with prompt caching
  of the system prompt and the tool list.
* ``scripted``: a fake model that replays fixed steps (tests, dry runs). It
  makes no network call.

The conversation uses a neutral message format:

* ``{"role": "system" | "user", "content": str}``
* ``{"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}]}``
* ``{"role": "tool", "tool_call_id": str, "name": str, "content": str}``

An endpoint error is retried (HTTP 408, 409, 425, 429 and 5xx, network
errors and timeouts) and then raises :class:`EndpointError`. An endpoint
error whose message shows that the endpoint could not parse the tool call
of the model (for example "error parsing tool call" or "syntax error") is
not retried: it raises :class:`ToolCallParseError`, a model format failure.

Keys come only from environment variables (the config gives the name of the
variable). The adapters never log request headers or keys. Endpoint URLs and
model names come from the private config, not from this code.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

ANTHROPIC_VERSION = "2023-06-01"
RETRY_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 529}
#: Endpoint error messages that show that the endpoint could not parse the tool call of the model
#: (case-insensitive). Such an error is a model format failure, not an endpoint failure.
TOOL_CALL_PARSE = re.compile(r"syntax error|tool[ _-]?calls? pars|pars(?:e|ed|ing) (?:the )?tool[ _-]?calls?", re.I)


class EndpointError(RuntimeError):
    """A model request failed after the retries (HTTP or network failure)."""


class ToolCallParseError(EndpointError):
    """The endpoint could not parse the tool call of the model (for example HTTP 500 "error parsing tool call").

    This is a model format failure. The adapter does not retry the request.
    """


def is_tool_call_parse_error(message: str) -> bool:
    """Return True if an endpoint error message shows a tool-call parse failure."""
    return bool(TOOL_CALL_PARSE.search(message or ""))


class ConfigError(ValueError):
    """The model config is not valid."""


@dataclass
class ToolCall:
    """One tool call of the model.

    Attributes
    ----------
    id : str
        Call id (from the model, or made by the adapter).
    name : str
        Tool name.
    arguments : Any
        Arguments as the model gave them: a dict, or a string that may not
        be valid JSON.
    """

    id: str
    name: str
    arguments: Any


@dataclass
class ModelReply:
    """One model reply.

    Attributes
    ----------
    text : str
        Text of the reply.
    tool_calls : list of ToolCall
        Tool calls of the reply.
    usage : dict
        ``input_tokens``, ``output_tokens``, ``cache_read_tokens``,
        ``cache_write_tokens`` and ``estimated`` (True if the endpoint gave
        no usage and the counts are characters / 4).
    finish_reason : str or None
        Finish reason of the endpoint (``"length"`` and ``"max_tokens"`` mean
        that the output was cut).
    duration_s : float
        Wall time of the request, with retries [s].
    provider_model : str or None
        Model string that the endpoint returned.
    retries : int
        Number of retries.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    duration_s: float = 0.0
    provider_model: str | None = None
    retries: int = 0

    @property
    def truncated(self) -> bool:
        """True if the endpoint cut the output at the token limit."""
        return self.finish_reason in ("length", "max_tokens")


def estimate_tokens(obj: Any) -> int:
    """Estimate a token count as characters / 4 of the JSON text of *obj*."""
    return int(len(json.dumps(obj, default=str)) / 4) + 1


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Parse tool-call arguments.

    Returns
    -------
    tuple
        ``(arguments, problem)``: the argument dict, or None and a short
        reason if the arguments are not a JSON object.
    """
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"arguments are not valid JSON: {exc}"
        if isinstance(val, dict):
            return val, None
        return None, f"arguments must be a JSON object, got {type(val).__name__}"
    return None, f"arguments must be a JSON object, got {type(raw).__name__}"


class BaseAdapter:
    """Base class of the model adapters.

    Parameters
    ----------
    config : dict
        Model config (from the private config file).
    """

    name = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.model = str(self.config.get("model", ""))
        self.retries = int(self.config.get("retries", 2))

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
                 temperature: float = 0.0, seed: int | None = None, max_tokens: int = 4096,
                 timeout_s: float = 180.0) -> ModelReply:
        """Send one request and return the reply."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Return facts about the endpoint for the run manifest (no keys, no URLs)."""
        return {"adapter": self.name, "model": self.model}

    # -- helpers ---------------------------------------------------------------
    def _api_key(self) -> str | None:
        env_name = self.config.get("api_key_env")
        if not env_name:
            return None
        key = os.environ.get(str(env_name))
        if not key:
            raise ConfigError(f"the environment variable {env_name} is not set")
        return key

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str],
              timeout_s: float) -> tuple[dict[str, Any], float, int]:
        data = json.dumps(body).encode("utf-8")
        attempt = 0
        t0 = time.perf_counter()
        last = ""
        while True:
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - URL from the private config
                    out = json.loads(resp.read().decode("utf-8"))
                return out, time.perf_counter() - t0, attempt
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                    last += f": {detail[:300]}"
                except OSError:
                    pass
                if is_tool_call_parse_error(detail):
                    raise ToolCallParseError(last) from None
                if exc.code not in RETRY_HTTP:
                    raise EndpointError(last) from None
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt >= self.retries:
                raise EndpointError(f"request failed after {attempt + 1} attempts: {last}")
            attempt += 1
            time.sleep(min(2.0 * attempt, 10.0))


class OpenAIChatAdapter(BaseAdapter):
    """Adapter for the OpenAI chat-completions format.

    Config keys: ``base_url`` (required, ends before ``/chat/completions``),
    ``model`` (required), ``api_key_env`` (optional; a local Ollama needs no
    key), ``seed_supported`` (default True), ``max_tokens_field`` (default
    ``"max_tokens"``), ``extra_body`` (optional dict), ``ollama`` (True for
    a local Ollama: the manifest then gets its version and model digest).
    """

    name = "openai_chat"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if not self.config.get("base_url") or not self.model:
            raise ConfigError("openai_chat needs 'base_url' and 'model' in the config")
        self.base_url = str(self.config["base_url"]).rstrip("/")

    @staticmethod
    def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MCP tools to OpenAI function tools."""
        return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                  "parameters": t.get("inputSchema", {"type": "object"})}}
                for t in tools]

    @staticmethod
    def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert neutral messages to the OpenAI format."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                        "name": c["name"],
                        "arguments": c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"]),
                    }} for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def complete(self, messages, tools, *, temperature=0.0, seed=None, max_tokens=4096, timeout_s=180.0):
        """Send one chat-completions request and return the reply."""
        body: dict[str, Any] = {"model": self.model, "messages": self.convert_messages(messages),
                                "temperature": temperature}
        if tools:
            body["tools"] = self.convert_tools(tools)
        body[str(self.config.get("max_tokens_field", "max_tokens"))] = int(max_tokens)
        if seed is not None and self.config.get("seed_supported", True):
            body["seed"] = int(seed)
        body.update(self.config.get("extra_body", {}))
        key = self._api_key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        out, dt, retries = self._post(self.base_url + "/chat/completions", body, headers, timeout_s)
        del key, headers
        choices = out.get("choices") or []
        if not choices:
            raise EndpointError("the reply has no choices")
        msg = choices[0].get("message") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            calls.append(ToolCall(id=str(c.get("id") or f"call_{i}"), name=str(fn.get("name", "")),
                                  arguments=fn.get("arguments", "")))
        usage = out.get("usage") or {}
        if usage:
            details = usage.get("prompt_tokens_details") or {}
            u = {"input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                 "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                 "cache_read_tokens": int(details.get("cached_tokens", 0) or 0),
                 "cache_write_tokens": 0, "estimated": False}
        else:
            u = {"input_tokens": estimate_tokens([messages, tools]), "output_tokens": estimate_tokens(msg),
                 "cache_read_tokens": 0, "cache_write_tokens": 0, "estimated": True}
        return ModelReply(text=msg.get("content") or "", tool_calls=calls, usage=u,
                          finish_reason=choices[0].get("finish_reason"), duration_s=dt,
                          provider_model=out.get("model"), retries=retries)

    def describe(self) -> dict[str, Any]:
        """Return the model facts; for a local Ollama also its version and model digest."""
        d = super().describe()
        if self.config.get("ollama"):
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            try:
                with urllib.request.urlopen(root + "/api/version", timeout=10) as r:  # noqa: S310
                    d["ollama_version"] = json.loads(r.read().decode()).get("version")
                with urllib.request.urlopen(root + "/api/tags", timeout=10) as r:  # noqa: S310
                    for m in json.loads(r.read().decode()).get("models", []):
                        if m.get("name") in (self.model, self.model + ":latest"):
                            d["model_digest"] = m.get("digest")
            except (urllib.error.URLError, OSError, ValueError):
                d["ollama_version"] = None
        return d


class AnthropicMessagesAdapter(BaseAdapter):
    """Adapter for the Anthropic Messages API.

    Config keys: ``base_url`` (required, ends before ``/v1/messages``),
    ``model`` (required), ``api_key_env`` (required), ``cache`` (default
    True: ``cache_control`` on the system prompt and the last tool).
    """

    name = "anthropic_messages"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if not self.config.get("base_url") or not self.model or not self.config.get("api_key_env"):
            raise ConfigError("anthropic_messages needs 'base_url', 'model' and 'api_key_env' in the config")
        self.base_url = str(self.config["base_url"]).rstrip("/")

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MCP tools to Anthropic tools (with a cache mark on the last one)."""
        out = [{"name": t["name"], "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", {"type": "object"})} for t in tools]
        if out and self.config.get("cache", True):
            out[-1]["cache_control"] = {"type": "ephemeral"}
        return out

    @staticmethod
    def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Convert neutral messages to ``(system, messages)`` of the Messages API."""
        system = ""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"]
            elif m["role"] == "user":
                out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
            elif m["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls") or []:
                    args, _ = parse_arguments(c["arguments"])
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": args or {}})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return system, out

    def complete(self, messages, tools, *, temperature=0.0, seed=None, max_tokens=4096, timeout_s=180.0):
        """Send one Messages request and return the reply (the API takes no seed)."""
        system, msgs = self.convert_messages(messages)
        body: dict[str, Any] = {"model": self.model, "max_tokens": int(max_tokens), "messages": msgs,
                                "temperature": temperature}
        if system:
            sblock: dict[str, Any] = {"type": "text", "text": system}
            if self.config.get("cache", True):
                sblock["cache_control"] = {"type": "ephemeral"}
            body["system"] = [sblock]
        if tools:
            body["tools"] = self.convert_tools(tools)
        key = self._api_key()
        headers = {"x-api-key": key or "", "anthropic-version": ANTHROPIC_VERSION}
        out, dt, retries = self._post(self.base_url + "/v1/messages", body, headers, timeout_s)
        del key, headers
        text = ""
        calls = []
        for b in out.get("content") or []:
            if b.get("type") == "text":
                text += b.get("text", "")
            elif b.get("type") == "tool_use":
                calls.append(ToolCall(id=str(b.get("id")), name=str(b.get("name", "")), arguments=b.get("input", {})))
        usage = out.get("usage") or {}
        u = {"input_tokens": int(usage.get("input_tokens", 0) or 0),
             "output_tokens": int(usage.get("output_tokens", 0) or 0),
             "cache_read_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
             "cache_write_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
             "estimated": not bool(usage)}
        return ModelReply(text=text, tool_calls=calls, usage=u, finish_reason=out.get("stop_reason"),
                          duration_s=dt, provider_model=out.get("model"), retries=retries)


class ScriptedAdapter(BaseAdapter):
    """A fake model that replays fixed steps. It makes no network call.

    Parameters
    ----------
    config : dict
        Model config. ``model`` is the label; ``delay_s`` (optional) is a
        sleep per reply [s]; ``repeat_last`` (optional) repeats the last
        step when the script ends.
    steps : list, optional
        The steps. Each step is a dict ``{"text": str, "tool_calls":
        [{"name", "arguments"}], "usage": dict}`` or a callable
        ``(messages, tools) -> dict``. If None, the config key ``steps``
        is used.
    """

    name = "scripted"

    def __init__(self, config: dict[str, Any], steps: list[Any] | None = None):
        super().__init__(config)
        self.steps: list[dict[str, Any] | Callable[..., dict[str, Any]]] = list(
            steps if steps is not None else self.config.get("steps", []))
        self.delay_s = float(self.config.get("delay_s", 0.0))
        self.repeat_last = bool(self.config.get("repeat_last", False))
        self.n_requests = 0
        self.last_tools: list[dict[str, Any]] = []

    def complete(self, messages, tools, *, temperature=0.0, seed=None, max_tokens=4096, timeout_s=180.0):
        """Return the next step of the script."""
        t0 = time.perf_counter()
        self.last_tools = tools
        if self.delay_s:
            time.sleep(self.delay_s)
        i = self.n_requests
        self.n_requests += 1
        if i < len(self.steps):
            step = self.steps[i]
        elif self.repeat_last and self.steps:
            step = self.steps[-1]
        else:
            step = {"text": ""}
        if callable(step):
            step = step(messages, tools)
        calls = [ToolCall(id=f"call_{i}_{j}", name=c["name"], arguments=c.get("arguments", {}))
                 for j, c in enumerate(step.get("tool_calls", []))]
        usage = step.get("usage") or {"input_tokens": estimate_tokens([messages, tools]),
                                      "output_tokens": estimate_tokens(step), "cache_read_tokens": 0,
                                      "cache_write_tokens": 0, "estimated": True}
        return ModelReply(text=step.get("text", ""), tool_calls=calls, usage=usage,
                          finish_reason=step.get("finish_reason", "stop"), duration_s=time.perf_counter() - t0,
                          provider_model=self.model)


ADAPTERS = {"openai_chat": OpenAIChatAdapter, "anthropic_messages": AnthropicMessagesAdapter,
            "scripted": ScriptedAdapter}


def make_adapter(config: dict[str, Any]) -> BaseAdapter:
    """Make the adapter that the config names (key ``adapter``)."""
    kind = config.get("adapter")
    if kind not in ADAPTERS:
        raise ConfigError(f"unknown adapter {kind!r}; use one of {sorted(ADAPTERS)}")
    return ADAPTERS[kind](config)
