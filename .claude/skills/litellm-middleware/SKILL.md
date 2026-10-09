---
name: litellm-middleware
description: Use when adding, changing, or removing a LiteLLM proxy middleware in this repo (src/litellm_middleware/), adding a LiteLLM hook to the pipeline, or bumping the pinned LiteLLM image the integration tests run against. Not for LiteLLM model, MCP, or guardrail config, which lives in the cluster repo.
argument-hint: <middleware-name>
---

# Add a LiteLLM Middleware

The README's Layout, Testing and Deployment sections hold the rules. This skill gives the order of work. A middleware ships in two steps: a release of this repo, then a tag bump of the image volume in spruyt-labs.

## Paths

| Item              | Path                                                   |
| ----------------- | ------------------------------------------------------ |
| Middleware dir    | `src/litellm_middleware/<name>/`                       |
| Pipeline hooks    | `src/litellm_middleware/pipeline.py`                   |
| Registry          | `src/litellm_middleware/registry.py`                   |
| Unit tests        | `tests/unit/<name>/test_<name>.py`                     |
| Import test       | `tests/unit/test_production_imports.py`                |
| Integration tests | `tests/integration/test_proxy.py`                      |
| Pinned LiteLLM    | `litellm-image.yaml`                                   |

`<name>` is snake_case (`secret_masking`). The registry name is kebab-case (`secret-masking`).

## Workflow

1. **Issue.** Find or create the GitHub issue.
2. **Hook check.** Confirm that `MiddlewarePipeline` in `pipeline.py` already delegates the LiteLLM hook you need. If it doesn't, add the hook there first, with its own test in `tests/unit/test_pipeline.py`. LiteLLM only calls hooks the callback class defines itself, so the pipeline must define the method, not inherit it.
3. **Red.** Create `tests/unit/<name>/test_<name>.py`. Copy the fake-`litellm` fixture from `tests/unit/secret_masking/test_secret_masking.py` and import `litellm_middleware.<name>.<name>` (pytest already puts `src/` on the path). Run `uv run pytest` and watch it fail.
4. **Green.** Create `<name>/__init__.py` (empty) and `<name>/<name>.py`, ending in a module-level instance (`<name> = <Name>Middleware()`). Helpers used only by this middleware go in `<name>/`. Runtime imports must already ship in the LiteLLM image. Test-only dependencies go in the `dev` group in `pyproject.toml` (then run `uv lock`).
5. **Register.** Add a `MiddlewareSpec` to `DEFAULT_MIDDLEWARE_SPECS` in `registry.py`, pointing at `litellm_middleware.<name>.<name>`. Specs run top to bottom, so order matters. Default to `required=False`. Extend the registry tests to assert the spec and its `required` value.
6. **Production import.** Add the dotted module to `test_production_dotted_imports_resolve`, and add a test that the middleware lands in `pipeline_plugin.pipeline_middleware.middlewares`.
7. **Integration test.** Add a test to `tests/integration/test_proxy.py` that sends a real request through the proxy and asserts the middleware's effect. Extend `fake_upstream.py` if the upstream must return something new. Watch the test fail before you implement the change.
8. **Verify.**

   ```bash
   uv run ruff format . && uv run ruff check .
   uv run pytest
   LITELLM_IT_RUNNER=agent-run uv run pytest tests/integration   # Coder; plain `uv run pytest tests/integration` elsewhere
   ```

   No packaging step exists. The Dockerfile copies the whole package, and `test_package_image.py` fails if the image and `src/litellm_middleware/` differ.

9. **Docs.** If anything is non-obvious (an upstream workaround, a removal condition, a failure mode), add a `##` section to `README.md` and a row to its middleware table.
10. **Ship.** Open a PR with a conventional commit (`feat:` for a new middleware); its title becomes the squash commit. CI runs the unit and integration suites and builds the image. After it merges, release-please opens a release PR; merging that publishes `ghcr.io/anthony-spruyt/litellm-middleware:<version>`. Only `feat`, `fix`, `perf`, `refactor` and `revert` commits cut a release.
11. **Deploy.** In spruyt-labs, bump the litellm-middleware image volume reference in `cluster/apps/litellm/litellm/app/values.yaml` to the new tag. No other cluster change is needed: the mount path, `PYTHONPATH` and `callbacks:` value stay the same. After the rollout, check the litellm pod logs for `failed to load <registry-name> middleware` (the kebab-case name).

## Bumping LiteLLM

Renovate updates the tag and digest in `litellm-image.yaml`, and CI runs the integration suite on that PR. Once it merges, spruyt-labs' Renovate offers the cluster the same tag and digest. It reads them from this file on `main`, so the cluster cannot get ahead of what CI tested.

If the new LiteLLM needs a middleware change, don't push it to the bump PR. Merging both together would hand the cluster the new LiteLLM while it still runs the old middleware image. Land the change as its own `fix:` PR that works on both versions, release it, and deploy it to the cluster. Then re-run the bump PR's checks and merge it.

## Gotchas

| Symptom                                                | Cause                                                                            |
| ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `ModuleNotFoundError` in pod, tests pass               | Pod `PYTHONPATH` isn't the mount path, or the volume references an old tag       |
| Hook never fires in pod                                | Pipeline doesn't define that hook, or the spec isn't in the registry             |
| Optional middleware silently absent                    | Import failed; `required=False` only logs a warning                              |
| New pod never Ready after rollout                      | A `required=True` middleware failed to import (by design)                        |
| Tests pass alone, fail together                        | Module cache: pop the `litellm_middleware.<name>.*` keys from `sys.modules` in the fixture |
| Integration: `LiteLLM container exited during startup` | Import error in the package; the failure message includes the proxy log          |
| `test_image_ships_only_the_package_sources` fails      | `.dockerignore` lets a non-package file in, or the Dockerfile `COPY` changed     |
| Unit green, integration red                            | Real LiteLLM calls the hook differently from the stub; trust integration         |

## Removing a middleware

Reverse steps 5 to 7 and 9: drop the spec, its tests and the README section, then delete `src/litellm_middleware/<name>/` and `tests/unit/<name>/`. Release, then bump the tag in spruyt-labs.
