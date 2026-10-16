# litellm-middleware

Composable middleware for the [LiteLLM](https://github.com/BerriAI/litellm) proxy. LiteLLM loads one callback, `MiddlewarePipeline`. The pipeline runs each middleware listed in `registry.py` on every LiteLLM hook it defines.

| Middleware          | Required | What it does                                                                                           |
| ------------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| `secret-masking`    | yes      | Swaps credentials for same-shape fakes before the request leaves the proxy, and back in the reply      |
| `ratelimit-headers` | no       | Restores Anthropic's `anthropic-ratelimit-unified-*` response headers that LiteLLM renames             |
| `tool-guard`        | no       | Marks tool results a scanner flags, or could not check, as untrusted data, for configured keys         |
| `fallback-auth`     | no       | Drops client credentials from calls to `anthropic-api/*` model groups, so they use their own `api_key` |

A required middleware that fails to import stops the proxy from starting. An optional one logs `failed to load <name> middleware` and the proxy serves without it.

## Layout

```text
src/litellm_middleware/        the package; the only thing the image ships
  base.py  pipeline.py  pipeline_plugin.py  registry.py
  <name>/
    __init__.py                empty
    <name>.py                  module-level instance that registry.py loads
    <helper>.py                used by this middleware only
tests/
  unit/                        mirrors the package; litellm is stubbed
  integration/                 boots the real LiteLLM image
litellm-image.yaml             the pinned LiteLLM image
postgres-image.yaml            the pinned Postgres image for the integration tests
Dockerfile                     package-only image
```

- `<name>` is the snake_case form of the registry name (`secret-masking` → `secret_masking/`).
- Import the core with `from ..pipeline import ...` and helpers with `from .<helper> import ...`.
- Runtime dependencies must already ship in the LiteLLM image. The `dev` dependency group only pins them so the unit tests can import them.

## Testing

```bash
uv run pytest                                  # unit
LITELLM_IT_RUNNER=agent-run uv run pytest tests/integration   # integration, Coder workspace
uv run pytest tests/integration                # integration, anywhere `docker run --rm` works
uv run ruff check . && uv run ruff format --check .
```

- **Unit** tests run each middleware against a stubbed `litellm`. They are fast, but they can't catch LiteLLM renaming a hook or changing its signature or call order.
- **Integration** tests need a container runtime. They build the `Dockerfile`, unpack the image's files, and mount them read-only into the pinned LiteLLM image at the same path and with the same `PYTHONPATH` as production. A fake upstream (Anthropic and OpenAI formats) and a fake tool-guard scanner run in the same container. Each test sends a real request and checks what reached the upstream and what came back.
- CI runs both suites on Python 3.13, the LiteLLM image's interpreter (`.python-version`), through repo-operator's shared `_build-image.yaml`, then builds the image. The GitHub runner's Docker serves the integration suite.
- `test_package_image.py` asserts that the image holds exactly the package sources and that LiteLLM imports the callback from the mount.
- `LITELLM_IT_RUNNER` overrides the `docker run --rm` prefix. `LITELLM_IT_CLI` overrides the CLI used for `build`, `exec` and `rm`, and it must share the runner's container store. Set `LITELLM_IT_SHOW_LOGS=1` to print the proxy log.
- The test proxy only sets the `general_settings` keys the tests depend on (`include_call_id_in_error_body`, `store_prompts_in_spend_logs`, a short spend-log write interval). It has a Postgres container, started and stopped by the fixture from the pinned `postgres-image.yaml` image (a ghcr image, since CI egress allows only ghcr), so tests can read spend logs and the Guardrails Monitor endpoints. It has no Redis or other network. A test-only `custom_auth` (`fake_auth.py`, mode `auto`) gives one key an alias; every other key goes through LiteLLM's own database auth.
- `count_tokens` ignores `api_base` and always calls `api.anthropic.com`, so its integration test calls the wrapped `_try_provider_token_count` inside the container instead of going over HTTP.

## Deployment

Each release publishes `ghcr.io/anthony-spruyt/litellm-middleware:<version>`, a `FROM scratch` image (about 75 kB) whose only content is `/litellm_middleware/`. Mount it as a read-only Kubernetes [image volume](https://kubernetes.io/docs/concepts/storage/volumes/#image) and point `PYTHONPATH` at the mount:

| Setting                        | Value                                                     |
| ------------------------------ | --------------------------------------------------------- |
| Mount path                     | `/opt/litellm-middleware`                                 |
| `PYTHONPATH`                   | `/opt/litellm-middleware`                                 |
| `litellm_settings.callbacks`   | `litellm_middleware.pipeline_plugin.pipeline_middleware`  |

```yaml
spec:
  containers:
    - name: litellm
      env:
        - name: PYTHONPATH
          value: /opt/litellm-middleware
      volumeMounts:
        - name: litellm-middleware
          mountPath: /opt/litellm-middleware
          readOnly: true
  volumes:
    - name: litellm-middleware
      image:
        reference: ghcr.io/anthony-spruyt/litellm-middleware:<version>
        pullPolicy: IfNotPresent
```

Image volumes need a cluster and container runtime that support them. Without them, copy `src/litellm_middleware/` into a directory on `PYTHONPATH`.

After a rollout, check the LiteLLM logs for `failed to load <registry-name> middleware`.

CI builds and publishes the image with outbound network limited to GitHub, Sigstore, the container registries and the hosts listed under `egress` in `metadata.yaml`.

## LiteLLM version coupling

The middleware hooks into LiteLLM internals: hook names and signatures, `_hidden_params`, and `proxy_server._try_provider_token_count`. These change between LiteLLM releases without notice.

`litellm-image.yaml` pins the LiteLLM image (tag and digest) the integration tests run against. Renovate tracks it through its `# renovate:` annotation, so a LiteLLM bump PR here runs the integration suite against the new version.

spruyt-labs takes its LiteLLM version and digest from this file on `main`, not from the registry, so the cluster can only move to a LiteLLM that passed CI here. Don't rename or move the file, or reshape its `image` value, without updating the `litellm-middleware-tested` custom datasource in spruyt-labs' `renovate-overrides.json5`. It sits at the repo root because Renovate's `config:recommended` ignores `**/tests/**`.

`pyproject.toml` and `uv.lock` aren't checked: the image doesn't ship them.

## Rate-limit headers

LiteLLM renames every non-OpenAI upstream header to `llm_provider-<name>` and has no setting to turn that off. As a result Claude Code never sees `anthropic-ratelimit-unified-*`, and its status line gets `rate_limits: null`. `ratelimit_headers/` adds un-prefixed copies of that header family only, from `async_post_call_response_headers_hook`, and leaves the prefixed ones in place.

- Streamed replies: the headers come from `response._hidden_params["additional_headers"]`. Non-streamed replies: the proxy pops `_hidden_params` from dict responses before the hook runs, so the raw upstream headers come from `data["litellm_logging_obj"].model_call_details["httpx_response"]`.
- LiteLLM only calls a hook if the callback's own class defines it (a leaf `__dict__` check). `MiddlewarePipeline` must therefore define every hook it delegates, not inherit it.
- Not covered: error replies (429s) and the opt-in `LITELLM_RUST` `/v1/messages` path, which sets neither header source.
- Remove it once LiteLLM forwards `anthropic-ratelimit-unified-*` unprefixed or adds a setting to do so.

## Fallback auth

LiteLLM attaches the client's `Authorization` header to the request before routing, and for a `Bearer sk-ant-oat...` token it overrides the deployment's `api_key`. A router fallback reuses those request kwargs, so an `anthropic-api/<model>` fallback deployment would send the client's token instead of its own key. `fallback_auth/` runs from `async_pre_call_deployment_hook`, after the deployment is chosen, and for deployments whose model name starts with `anthropic-api/` removes `authorization`, `x-api-key` and the `oauth-*` tokens of `anthropic-beta` (the header goes if none are left).

- The deployment's model name is `deployment_model_name`, which the router writes for each attempt into the bucket that matches the call type: `kwargs["litellm_metadata"]` for `anthropic_messages`, `aanthropic_messages`, `aresponses` and `responses` (`/v1/messages`, `/v1/responses`), and `kwargs["metadata"]` for `acompletion`, `completion`, `atext_completion` and `text_completion` (`/v1/chat/completions`). Only that bucket is read, and the requested `model_group` plays no part. The headers come from `kwargs["headers"]`, `kwargs["extra_headers"]` and the `extra_headers` of each `kwargs["provider_specific_header"]` entry.
- It returns copies, so the kwargs the router reuses for the next attempt are unchanged.
- If scrubbing raises for an `anthropic-api/*` deployment, all three header containers (`headers`, `extra_headers`, `provider_specific_header`) are dropped from the returned copy. Both that and a failed deployment lookup log a warning with the exception type name, never header values.
- `anthropic-beta` values given as `str`, `bytes`, `list` or `tuple` are scrubbed and keep their type.
- Because the pipeline defines `async_pre_call_deployment_hook`, the synchronous `litellm.responses` path also takes the hook branch (`_run_sync_responses_pre_call_deployment_hook` in `llm_http_handler.py`) and the request is unchanged.
- Remove it once LiteLLM stops letting a forwarded client token override a deployment's `api_key`.

## Secret masking

`secret_masking/` masks `/v1/messages`, `/v1/chat/completions`, `/v1/responses`, `/v1/responses/compact`, `/v1/completions`, Gemini `generateContent`/`streamGenerateContent`, and the body that `count_tokens`/`input_tokens` forward to the provider.

It swaps credentials with a known prefix (GitHub, Google, Anthropic, OpenAI, AWS, Cloudflare, Context7, SendGrid, Shopify, DigitalOcean, Perplexity, Postman, Linear, Notion, LiteLLM `sk-`, `sl_`, PEM private keys and the others in `_PATTERNS`) for a fake with the same prefix, length and character classes.

It swaps fakes in the reply back to the real values, including streamed text and tool-call arguments.

- Fakes are an HMAC of the real value keyed from `LITELLM_SALT_KEY`, so one secret gets the same fake across turns and replicas, and prompt caching still hits. If the salt is missing, each pod logs a warning and uses a random key.
- Only prefixed formats are caught. Bare high-entropy strings (hashes, UUIDs, unprefixed passwords) pass through so commit SHAs and similar values aren't mangled.
- Thinking and reasoning blocks, base64 sources, `data:` URLs, `input_audio` and remote image/file URLs are never touched.
- The fake-to-real map is kept per virtual key for an hour, so a fake the model echoes from an earlier turn is still swapped back. Each pod keeps its own map in memory (capped at 2000 fakes).
- `shared_fakes.py` shares the map through Redis/Valkey (needs `HSETEX`, Valkey 9+) when both `REDIS_HOST` and `LITELLM_SALT_KEY` are set (`REDIS_PORT`, `REDIS_USERNAME` and `REDIS_PASSWORD` are optional). Entries are AES-GCM encrypted with a key derived from `LITELLM_SALT_KEY`. Sharing is best effort: if the store is unreachable, the pod restores from its own memory.

## Tool guard

`tool_guard/` sends tool results to a scanner before the request goes upstream, and wraps each one the scanner flags, or that could not be checked, in a marker that names its verdict and tells the model the text is tool data, not instructions. It runs after `secret-masking`, so the scanner only sees masked text. It reads tool results from:

- `/v1/messages`: `tool_result` blocks in user turns, including nested `document` and `search_result` blocks, and server tool results (`web_fetch_tool_result`, `mcp_tool_result` and other `*_tool_result` blocks) in assistant turns
- `/v1/chat/completions`: `role: tool` and `role: function` messages
- `/v1/responses` and `/v1/responses/compact`: `function_call_output`, `custom_tool_call_output`, shell and patch call outputs, and `mcp_call` output
- Gemini `generateContent`: `functionResponse` parts

| Variable                     | Purpose                                                       | Default            |
| ---------------------------- | ------------------------------------------------------------- | ------------------ |
| `TOOL_GUARD_URL`             | Scanner base URL. Unset turns the middleware off              | unset              |
| `TOOL_GUARD_KEY_ALIASES`     | Comma-separated virtual key aliases to scan for               | none               |
| `TOOL_GUARD_TEAM_IDS`        | Comma-separated team IDs to scan for                          | none               |
| `TOOL_GUARD_TIMEOUT_SECONDS` | Time limit for all scanner calls on one request               | `3`                |
| `TOOL_GUARD_MAX_TEXT_BYTES`  | Size above which a tool result is wrapped without a scan      | `262144` (256 KiB) |
| `TOOL_GUARD_MAX_BATCH_BYTES` | Size limit for one scanner request body                       | `4194304` (4 MiB)  |

A request is scanned when its key alias or team ID is listed. Enforced keys can use the endpoints listed above, `/v1/messages/count_tokens`, embeddings, moderations, image generation, and tool calls through LiteLLM's MCP gateway. Any other endpoint, Responses WebSocket mode included, answers their requests with a client error.

- Each tool result is identified by `sha256:` plus the hex SHA-256 of its text: the string content, or its text fields joined in order with newlines. For Gemini `functionResponse`, the text fields are the object keys and string values of `response`.
- Results after the last assistant turn, and server tool results in the last assistant turn, go to the scanner with their text. Older results go by hash only, so the scanner answers from its verdict cache.
- A tool result larger than `TOOL_GUARD_MAX_TEXT_BYTES` (UTF-8) is marked `too-big` without a scanner call.
- Scanner requests are split into batches whose JSON body is at most `TOOL_GUARD_MAX_BATCH_BYTES`, so the body limit is counted on the encoded text (quotes and control characters take 2 to 6 bytes). Batches are sent one after another, newest results first, inside the same time limit. A tool result whose encoded size alone exceeds `TOOL_GUARD_MAX_BATCH_BYTES` is marked `unchecked` without a scanner call. Keep the value below the scanner's request body limit.
- The marker is `<untrusted-tool-output id="..." verdict="...">` ... `</untrusted-tool-output id="...">`, where the id is the first 16 hex digits of the result's hash, so the text inside cannot contain its own closing tag.
- The verdict is `suspected-prompt-injection` for results the scanner flags, `too-big` for results over `TOOL_GUARD_MAX_TEXT_BYTES`, and `unchecked` for results without a verdict. A header line after the opening tag tells the model how to treat each: flagged and too-big output is data only, and the model tells the user if its task depends on flagged output.
- The rewrite changes only the text of marked results, and the same input and verdicts always give the same output, so prompt caching keeps working.
- When a scanner call fails or runs out of time, every result in the request without a verdict from a completed call is marked `unchecked`. Errors and timeouts are logged as `tool guard scan error: <error>`.

Scanner contract:

```text
POST {TOOL_GUARD_URL}/v1/scan
{"new": [{"hash": "sha256:...", "text": "..."}], "known": ["sha256:..."]}
-> {"flagged": ["sha256:..."], "unknown": ["sha256:..."], "scores": {"sha256:...": 0.97}}
```

`unknown` lists `known` hashes the scanner has no verdict for. The middleware sends those results again with their text in follow-up calls, inside the same time limit, and wraps anything a call flags. Results in a call that fails, and in calls not sent before the time limit, are marked `unchecked`.

`scores` is optional: a score for each flagged hash, as a finite number. A flagged hash without a usable score is recorded as a `scanner-error`, and the result is still wrapped.

### LiteLLM UI record

A request with at least one new tool result (one after the last assistant turn) gets one standard LiteLLM guardrail entry named `tool-guard`, so the verdicts show in the request's Logs drawer and in the Guardrails Monitor. Requests without a new tool result get none. Recording never fails the request: an error is logged as `tool guard record error: <type>` and skipped.

| Status                        | When                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `guardrail_flagged`           | The scanner flagged a new result and gave it a score                                                    |
| `guardrail_failed_to_respond` | No result is flagged with a score, and a new result is `unchecked`, `too-big` or a flag without a score |
| `success`                     | Every new result was scanned and none was flagged                                                       |

The entry's `confidence_score` is the highest score among the new flagged results. `guardrail_intervened` is never used, because the middleware wraps text and does not block.

The Logs drawer shows the entry as a Match Details table with one row per distinct new result that has a verdict. Each row has a Type, Method, Action and Detail:

| Column | Flagged result                                                    | `unchecked`, `too-big` or `scanner-error` result |
| ------ | ----------------------------------------------------------------- | ------------------------------------------------ |
| Type   | The tool name, or `tool` when the request does not name it        | Same                                             |
| Method | `prompt-guard`                                                    | Same                                             |
| Action | `FLAG`                                                            | `SKIP`                                           |
| Detail | `[score]` and `<tool input> → <tool output>`, the score to 2 places | `[verdict]` and the tool input                   |

The action is never `BLOCK`, because the middleware wraps text and does not block. The tool input and the output are each cut to 300 bytes (head and tail around a `...[N bytes cut]...` marker). The input is read from the matching call in the request: the `input` of an Anthropic `tool_use`, `server_tool_use` or `mcp_tool_use` block, a chat tool call's `function.arguments`, a Responses call's `arguments` (`input` for `custom_tool_call`), and a Gemini `functionCall`'s `args`, matched by call ID, or by name when the latest model turn called that name once. An input object with only a `command` string shows as that command; any other object shows as compact JSON. When the request does not give the input, the Detail shows only the output (flagged result) or is empty (skipped result).

A flagged entry also sets:

- `detection_method` to `prompt-guard`
- `classification` to `{category: "suspected-prompt-injection", confidence, reason}`, where `confidence` is the highest score and `reason` names its tool, for example `Bash output scored 0.97`
- `violation_categories` to `["suspected-prompt-injection"]`
- `guardrail_action` to `FLAGGED`

An entry with no flagged result sets only `detection_method`.

`guardrail_response` holds a `summary` string and `counts`, the number of distinct new results per verdict (`suspected-prompt-injection`, `scanner-error`, `unchecked` and `too-big`). The whole record is kept under 64 KiB: past that, trailing rows are dropped and counted in `omitted`.

LiteLLM replaces `guardrail_response`, `match_details` and `classification` with `REDACTED_BY_LITELM` in the spend logs unless `general_settings.store_prompts_in_spend_logs` is `true`. Without it the Logs drawer still shows the status, `confidence_score`, `detection_method`, `violation_categories` and `guardrail_action`, but not the summary, the Match Details rows or the classification. The setting also stores request and response bodies in the spend logs.

The Guardrails Monitor opens a per-guardrail detail page from the overview row, and that page only loads for a guardrail LiteLLM has registered. `tool_guard/placeholder.py` ships `ToolGuardPlaceholder`, a `CustomGuardrail` with no hooks, so it never blocks, changes or records anything, even if a caller opts in with `guardrails: ["tool-guard"]`. The record still comes from the middleware. Register it in the proxy config; the dotted path resolves through `PYTHONPATH`, so it works with the mount described under Deployment:

```yaml
guardrails:
  - guardrail_name: tool-guard
    litellm_params:
      guardrail: litellm_middleware.tool_guard.placeholder.ToolGuardPlaceholder
      mode: pre_call
      default_on: false
```

The `guardrail_name` must stay `tool-guard`: the overview row, the detail page and the logs list match the record by that name. Without the entry the overview row and Logs drawer still work, but the detail page shows "Failed to load guardrail details".
