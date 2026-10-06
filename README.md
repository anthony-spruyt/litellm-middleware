# litellm-middleware

Composable middleware for the [LiteLLM](https://github.com/BerriAI/litellm) proxy. LiteLLM loads one callback, `MiddlewarePipeline`. The pipeline runs each middleware listed in `registry.py` on every LiteLLM hook it defines.

| Middleware          | Required | What it does                                                                                         |
| ------------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `secret-masking`    | yes      | Swaps credentials for same-shape fakes before the request leaves the proxy, and back in the reply    |
| `ratelimit-headers` | no       | Restores Anthropic's `anthropic-ratelimit-unified-*` response headers that LiteLLM renames           |

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
    litellm.yaml               the pinned LiteLLM image
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
- **Integration** tests need a container runtime. They build the `Dockerfile`, unpack the image's files, and mount them read-only into the pinned LiteLLM image at the same path and with the same `PYTHONPATH` as production. A fake Anthropic upstream runs in the same container. Each test sends a real request and checks what reached the upstream and what came back.
- CI runs both suites on Python 3.13, the LiteLLM image's interpreter (`.python-version`), through repo-operator's shared `_build-image.yaml`, then builds the image. The GitHub runner's Docker serves the integration suite.
- `test_package_image.py` asserts that the image holds exactly the package sources and that LiteLLM imports the callback from the mount.
- `LITELLM_IT_RUNNER` overrides the `docker run --rm` prefix. `LITELLM_IT_CLI` overrides the CLI used for `build`, `exec` and `rm`, and it must share the runner's container store. Set `LITELLM_IT_SHOW_LOGS=1` to print the proxy log.
- The test proxy only sets the `general_settings` keys the middleware depends on (`include_call_id_in_error_body`). It has no database, Redis or network.
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

`tests/integration/litellm.yaml` pins the LiteLLM image (tag and digest) the integration tests run against. Keep it equal to the version the cluster deploys, and bump both together. Renovate tracks the pin through its `# renovate:` annotation, so a LiteLLM bump PR here runs the integration suite against the new version before the cluster takes it.

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
