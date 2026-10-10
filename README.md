# litellm-middleware

Composable middleware for the [LiteLLM](https://github.com/BerriAI/litellm) proxy. LiteLLM loads one callback, `MiddlewarePipeline`. The pipeline runs each middleware listed in `registry.py` on every LiteLLM hook it defines.

| Middleware          | Required | What it does                                                                                         |
| ------------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `secret-masking`    | yes      | Swaps credentials for same-shape fakes before the request leaves the proxy, and back in the reply    |
| `ratelimit-headers` | no       | Restores Anthropic's `anthropic-ratelimit-unified-*` response headers that LiteLLM renames           |
| `tool-guard`        | no       | Marks tool results a scanner flags, or could not check, as untrusted data, for configured keys       |

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
- The test proxy only sets the `general_settings` keys the middleware depends on (`include_call_id_in_error_body`). It has no database, Redis or network. A test-only `custom_auth` (`fake_auth.py`, mode `auto`) gives one key an alias; every other key falls through to the master key.
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

## Secret masking

`secret_masking/` masks `/v1/messages`, `/v1/chat/completions`, `/v1/responses`, `/v1/responses/compact`, `/v1/completions`, Gemini `generateContent`/`streamGenerateContent`, and the body that `count_tokens`/`input_tokens` forward to the provider.

It swaps credentials with a known prefix (GitHub, Google, Anthropic, OpenAI, AWS, LiteLLM `sk-`, `sl_`, PEM private keys and the others in `_PATTERNS`) for a fake with the same prefix, length and character classes.

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

A request is scanned when its key alias or team ID is listed. Enforced keys can use the endpoints listed above, `/v1/messages/count_tokens`, embeddings, moderations, image generation, and tool calls through LiteLLM's MCP gateway. Any other endpoint, Responses WebSocket mode included, answers their requests with a client error.

- Each tool result is identified by `sha256:` plus the hex SHA-256 of its text: the string content, or its text fields joined in order with newlines. For Gemini `functionResponse`, the text fields are the object keys and string values of `response`.
- Results after the last assistant turn, and server tool results in the last assistant turn, go to the scanner with their text. Older results go by hash only, so the scanner answers from its verdict cache.
- A tool result larger than `TOOL_GUARD_MAX_TEXT_BYTES` (UTF-8) is marked `unchecked` without a scanner call.
- The marker is `<untrusted-tool-output id="..." verdict="...">` ... `</untrusted-tool-output id="...">`, where the id is the first 16 hex digits of the result's hash, so the text inside cannot contain its own closing tag.
- The verdict is `suspected-prompt-injection` for results the scanner flags and `unchecked` for results without a verdict. A header line after the opening tag tells the model how to treat each: flagged output is data only, and the model tells the user if its task depends on it.
- The rewrite changes only the text of marked results, and the same input and verdicts always give the same output, so prompt caching keeps working.
- When a scanner call fails or runs out of time, every result in the request without a verdict from it is marked `unchecked`. Errors and timeouts are logged as `tool guard scan error: <error>`.

Scanner contract:

```text
POST {TOOL_GUARD_URL}/v1/scan
{"new": [{"hash": "sha256:...", "text": "..."}], "known": ["sha256:..."]}
-> {"flagged": ["sha256:..."], "unknown": ["sha256:..."]}
```

`unknown` lists `known` hashes the scanner has no verdict for. The middleware sends those results again with their text in a second call, inside the same time limit, and wraps anything either call flags. If the second call fails, the results it carried are marked `unchecked`.
