# Agent benchmark of the Ventorum MCP interface

This folder holds a harness that measures how well an LLM agent uses the Ventorum tools through the Model Context Protocol (MCP). It runs a public task set through the Ventorum MCP server, scores each run strictly, and writes JSON Lines logs and a summary to a folder outside the repository.

This file tells you how to run the harness. It holds no results.

## Parts

| File | Content |
| --- | --- |
| `tasks_v1.json` | Task set v1: 49 task templates (prompt, answer keys, accepted tools, golden calls, tolerances, accepted outcomes) with the seed-0 parameters and a canary line. It holds no reference values. |
| `tasks.py` | Load and check the task file, make variants from a seed, build the prompts. |
| `references.py` | Run the golden calls at run time, and the self-test against the library API. |
| `mcp_client.py` | Stdio MCP client: starts `python -m ventorum.agent --mcp`, sends `initialize`, `tools/list` and `tools/call`. |
| `adapters.py` | Model adapters `openai_chat` (OpenAI, Ollama, Gemini and other compatible endpoints), `anthropic_messages`, and `scripted` (a fake model for tests). |
| `scoring.py` | Final-answer extraction, tolerances, provenance check, geometry comparison, error kinds. |
| `runner.py` | Command line: `run`, `report`, `references`, `selftest`. |
| `client_track.py` | Track B: opencode and Claude Code driven by scripts, scored from the final text and the server audit log. |

The harness uses only the Python standard library and Ventorum.

## Rules

- Give the output folder with `--out`. It must be outside the repository; the harness refuses a folder inside it (also by another spelling of the same folder) and a folder inside any git checkout.
- Keep the model config outside the repository. It gives the endpoints, the model strings, the names of the key variables and the prices.
- Keys come only from environment variables. The harness never writes a key, a host name or a user name to an output file. A redaction filter removes strings that look like keys, and the values of all variables that the config names in `api_key_env`, from every record and from the audit logs. The MCP server does not get these variables.
- The harness refuses a model on the owner blacklist.
- Run one benchmark at a time on a machine (no concurrent benchmarks). `--jobs` above 1 is only for API models, and the run is then marked "times not comparable".

## Self-test

The self-test checks the task file, the arguments of every golden call, and six golden calls against the library API (`vt.analyze`, `vt.static_derivatives`, `vt.trim`, `vt.derivative_matrix`, `analyze_ground_effect`, `vt.numerical_error_bars`) within 0.5 %:

```bash
python -m validation.agent_benchmark.runner selftest
```

## Model config

The config is a JSON file. Example structure (all values are placeholders):

```json
{
  "models": {
    "local-model": {
      "adapter": "openai_chat",
      "base_url": "<endpoint URL, up to /v1>",
      "model": "<model name>",
      "local": true,
      "ollama": true,
      "context_length": 65536,
      "temperature": 0,
      "seed": 0
    },
    "api-model": {
      "adapter": "anthropic_messages",
      "base_url": "<endpoint URL>",
      "model": "<model string>",
      "api_key_env": "<NAME_OF_THE_KEY_VARIABLE>",
      "prices_usd_per_mtok": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}
    }
  },
  "limits": {"max_turns": 12, "max_tool_calls": 10, "cost_cap_usd": 20}
}
```

Model keys:

- `adapter`: `openai_chat`, `anthropic_messages` or `scripted`.
- `base_url`, `model`: endpoint and model string. `api_key_env`: the name of the environment variable that holds the key (not the key).
- `local`: true for a local model (longer default time limits, no `--jobs`).
- `context_length`: the context of the model in tokens. When a prompt fills it, the run gets `context_overflow` and is excluded from the success rate.
- `prices_usd_per_mtok`: price per million tokens, for the cost estimate and the cost cap.
- `seed_supported`, `max_tokens_field`, `extra_body`, `cache`: adapter options (see `adapters.py`).

## Run

```bash
python -m validation.agent_benchmark.runner run --config <config.json> --model local-model \
    --condition focused --tasks AB-01,AB-19,AB-25,AB-43 --repeats 1 --out <folder outside the repository>
```

- `--condition full` gives the model all 12 tools as the server serves them. `--condition focused` gives the task tools plus two fixed distractor tools.
- `--tasks` takes `all`, `track_b`, `level:N`, or ids and ranges (`AB-01,AB-05:AB-09`).
- `--repeats` is k (default 3).
- `--seed N` makes the held-out variants of seed N (0 is the public set).
- `--run-id ID` resumes the run `ID`: the finished task runs are skipped. A task run that ended with `endpoint_error` runs again, and the report keeps only its last record.
- The limits (`--max-turns`, `--max-tool-calls`, `--task-time-s`, `--tool-time-s`, `--request-time-s`, `--max-output-tokens`, `--cost-cap-usd`) override the config. The cost cap must be above 0 USD. At the cap the run stops cleanly; the cost of the task run that reached the cap is recorded (`run_stopped` event), and a resume adds it to the cost so far.

Each task run gets a new MCP server process, a hidden warm-up call before the task clock starts, a sandbox copy of the tuning profile folder (`VENTORUM_CONFIG_DIR`), an audit log (`VENTORUM_AGENT_AUDIT_LOG`) and an empty working folder.

At the start of a run the harness runs every golden call once and keeps the reference values in `references.json`. If an earlier run of the same Ventorum commit and task file exists in the output folder (or `--baseline` gives a stored set), the harness compares the values at 1e-4 relative and stops if one changed.

## Outputs

One folder per run, `<out>/<run_id>/`:

| File | Content |
| --- | --- |
| `manifest.json` | Schema and harness versions, task-set hash, system-prompt hash, Ventorum version and commit, Python, numpy and torch versions, OS, model facts, condition, temperature, seed, limits, warm-up device and precision. |
| `references.json` | Reference values of the golden calls on this machine. |
| `events.jsonl` | One record per event: model replies with usage and duration, tool calls with arguments, status, error type, payload and execution time. |
| `results.jsonl` | One scored record per task and repeat: status, values, per-value result, success, error kinds, tokens, times, cost. |
| `audit/` | The server audit log of each task run. |
| `summary.md`, `summary.csv` | Made by `report` from `results.jsonl`. |

```bash
python -m validation.agent_benchmark.runner report <out>/<run_id>
```

## Scoring

The model must end its last message with one block:

```
<final_answer>
{"status": "answered", "values": {"KEY": 0.0}, "caveats": []}
</final_answer>
```

A task run succeeds when the status, the values (within the task tolerances), the required caveat codes, the provenance of each number (a tool result or the task formula), and the argument checks agree with one accepted outcome of the task. A placeholder value (`"n/a"`, `""`, `"unknown"`, `"none"`, `"null"`, `"na"`, `"-"`) counts as no value. A value with two numbers (a range such as `"3 to 5"`) is not one value: it gets `format_error`. A task can name flight-condition entries that the geometry comparison does not compare (`ignore_condition`, for example the start angle of attack of a trim). The record also gives the error kinds (for example `wrong_units`, `sign_error`, `geometry_drift`, `hallucinated_value`, `unsupported_value`, `missed_refusal`, `false_refusal`, `format_error`, `limit:turns`, `loop:repeat_call`). The lenient score (numbers found in free text) is reported apart and never ranks models.

Two harness rules stop a task run as a model failure. The record is not excluded from the success rate:

- Repeated call: when the model sends a tool call identical to an earlier call of the same task run whose result was a success (same tool, same arguments after JSON normalisation), the harness does not run the call. It stops the task run with the error kind `loop:repeat_call`. A repeat of a call whose result was an error runs again.
- Tool-call parse failure: when the endpoint returns an error whose message shows that it could not parse the tool call of the model (for example HTTP 500 with "error parsing tool call" or "syntax error", case-insensitive), the harness does not retry the request. The record gets `format_error` and the sub-kind `format_error:tool_call_parse`.

Other endpoint errors (other 5xx errors, network errors, timeouts) are retried. After the retries the record gets `endpoint_error` and is excluded from the success rate, as is `context_overflow`.

## Track B (existing clients)

```bash
python -m validation.agent_benchmark.client_track --config <config.json> --client claude_code \
    --model <model> --out <folder outside the repository>
```

The default command lines of the clients are in `client_track.py`. Check them against the installed client version and override them in the config (`clients.<name>.command`). The client runs in an empty temporary folder with only the Ventorum MCP server and its built-in shell, file and web tools off. The prompt goes to the client through its standard input (a template can also use `{prompt_file}`); a template with `{prompt}` is refused. The client runs in a new process group, and at the timeout the harness stops the client and all its child processes.

The harness replays the calls of the audit log through a separate MCP server with a sandbox config folder to get the payloads. It never replays `ventorum_tune_machine`: that call gets only its status from the audit line. The references are checked as in track A.
