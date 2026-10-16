"""Boots the pinned LiteLLM image with the package image's files mounted where the pod mounts them."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

import httpx
import pytest
import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
MOUNT_PATH = "/opt/litellm-middleware"
CALLBACK = "litellm_middleware.pipeline_plugin.pipeline_middleware"
MASTER_KEY = "it-master-key"
MODEL = "claude-it"
OPENAI_MODEL = "gpt-it"
SUBSCRIPTION_MODEL = "anthropic/test-model"
API_KEY_MODEL = "anthropic-api/test-model"
ALIASED_API_KEY_MODEL = "aliased-test-model"
FALLBACK_API_KEY = "sk-" + "ant-api03-" + "fallback-fake"
GUARDED_KEY = "it-tool-guard-key"
GUARDED_ALIAS = "it-tool-guard"
SCANNER_PORT = 8097
SCANNER_MAX_BODY_BYTES = 8192
TOOL_GUARD_MAX_BATCH_BYTES = 4096
STARTUP_TIMEOUT_S = 240
MCP_SERVER = "itmcp"
MCP_PORT = 8098
SPEND_WRITE_INTERVAL_S = 2


def _runner() -> list[str]:
    # agent-run on Coder workspaces; it already passes --rm.
    return shlex.split(os.environ.get("LITELLM_IT_RUNNER", "docker run --rm"))


def _cli() -> str:
    # exec and rm; must share the runner's container store.
    return os.environ.get("LITELLM_IT_CLI", "docker")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def litellm_image() -> str:
    return yaml.safe_load((REPO_ROOT / "litellm-image.yaml").read_text())["image"]


def postgres_image() -> str:
    return yaml.safe_load((REPO_ROOT / "postgres-image.yaml").read_text())["image"]


def export_package_image(dest: Path) -> None:
    """Builds the package image and unpacks its filesystem, so tests see exactly what the image ships."""
    archive = dest.with_suffix(".tar")
    build = subprocess.run(
        [_cli(), "build", "--output", f"type=tar,dest={archive}", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def _world_readable(root: Path) -> None:
    # The container runs as a remapped non-root uid, so host file ownership doesn't carry over.
    for path in [root, *root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def _config() -> str:
    return yaml.safe_dump(
        {
            "model_list": [
                {
                    "model_name": MODEL,
                    "litellm_params": {
                        "model": f"anthropic/{MODEL}",
                        "api_base": "http://127.0.0.1:8099",
                        "api_key": "it",
                    },
                },
                {
                    "model_name": SUBSCRIPTION_MODEL,
                    "litellm_params": {"model": "anthropic/test-model", "api_base": "http://127.0.0.1:8099"},
                },
                {
                    "model_name": API_KEY_MODEL,
                    "litellm_params": {
                        "model": "anthropic/test-model",
                        "api_base": "http://127.0.0.1:8099",
                        "api_key": FALLBACK_API_KEY,
                    },
                },
                {
                    "model_name": OPENAI_MODEL,
                    "litellm_params": {
                        "model": f"openai/{OPENAI_MODEL}",
                        "api_base": "http://127.0.0.1:8099/v1",
                        "api_key": "it",
                    },
                },
            ],
            "litellm_settings": {"callbacks": [CALLBACK]},
            "guardrails": [
                {
                    "guardrail_name": "tool-guard",
                    "litellm_params": {
                        "guardrail": "litellm_middleware.tool_guard.placeholder.ToolGuardPlaceholder",
                        "mode": "pre_call",
                        "default_on": False,
                    },
                }
            ],
            "router_settings": {
                "fallbacks": [{SUBSCRIPTION_MODEL: [API_KEY_MODEL]}],
                "model_group_alias": {ALIASED_API_KEY_MODEL: API_KEY_MODEL},
            },
            "general_settings": {
                "forward_client_headers_to_llm_api": True,
                "include_call_id_in_error_body": True,
                "master_key": MASTER_KEY,
                "proxy_batch_write_at": SPEND_WRITE_INTERVAL_S,
                # Without it LiteLLM replaces guardrail_response in spend logs with a redaction marker.
                "store_prompts_in_spend_logs": True,
                # Gives GUARDED_KEY a key alias without the database; other keys go through LiteLLM's own auth.
                "custom_auth": "fake_auth.user_api_key_auth",
                "custom_auth_settings": {"mode": "auto"},
            },
            # Down at startup so the proxy's tool mapping stays cold, like a DB-loaded server after a restart.
            "mcp_servers": {
                MCP_SERVER: {"url": f"http://127.0.0.1:{MCP_PORT}/mcp", "transport": "http", "allow_all_keys": True}
            },
        }
    )


@pytest.fixture(scope="session")
def package_files(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("package-image")
    export_package_image(dest)
    _world_readable(dest)
    return dest


@pytest.fixture(scope="session")
def proxy(package_files, tmp_path_factory):
    work = tmp_path_factory.mktemp("litellm-it")
    (work / "config.yaml").write_text(_config())
    for fake in ("fake_upstream.py", "fake_mcp.py", "fake_scanner.py", "fake_auth.py"):
        (work / fake).write_bytes((HERE / fake).read_bytes())
    _world_readable(work)

    name = f"litellm-it-{uuid.uuid4().hex[:8]}"
    port = _free_port()
    log_path = work / "proxy.log"
    database = Postgres(f"{name}-db", _free_port(), work / "postgres.log")
    database.start()
    # Foreground, not -d: --rm deletes a crashed container and its logs, so capture output ourselves.
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                *_runner(),
                "--name",
                name,
                # Same port both sides: WSL devcontainer podman runs host-network and ignores the mapping.
                "-p",
                f"127.0.0.1:{port}:{port}",
                "-e",
                f"PYTHONPATH={MOUNT_PATH}",
                "-e",
                f"TOOL_GUARD_URL=http://127.0.0.1:{SCANNER_PORT}",
                "-e",
                f"TOOL_GUARD_KEY_ALIASES={GUARDED_ALIAS}",
                "-e",
                f"TOOL_GUARD_MAX_BATCH_BYTES={TOOL_GUARD_MAX_BATCH_BYTES}",
                "-e",
                f"FAKE_SCANNER_MAX_BODY_BYTES={SCANNER_MAX_BODY_BYTES}",
                "-e",
                "ANTHROPIC_API_BASE=http://127.0.0.1:8099",
                "-e",
                "ANTHROPIC_API_KEY=it",
                "-e",
                f"DATABASE_URL={database.url}",
                "-v",
                f"{package_files}:{MOUNT_PATH}:ro",
                "-v",
                f"{work / 'config.yaml'}:/app/config.yaml:ro",
                "-v",
                f"{work / 'fake_upstream.py'}:/it/fake_upstream.py:ro",
                "-v",
                f"{work / 'fake_mcp.py'}:/it/fake_mcp.py:ro",
                "-v",
                f"{work / 'fake_scanner.py'}:/it/fake_scanner.py:ro",
                "-v",
                f"{work / 'fake_auth.py'}:/app/fake_auth.py:ro",
                "--entrypoint",
                "sh",
                litellm_image(),
                "-c",
                "python /it/fake_upstream.py & python /it/fake_scanner.py & "
                f"exec litellm --config /app/config.yaml --port {port}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    proxy = Proxy(f"http://127.0.0.1:{port}", name, log_path, database)
    try:
        _wait_ready(proxy, proc)
        yield proxy
    finally:
        if os.environ.get("LITELLM_IT_SHOW_LOGS"):
            print(proxy.logs())
        subprocess.run([_cli(), "rm", "-f", name], capture_output=True, check=False)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        database.stop()


def _wait_ready(proxy: Proxy, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{proxy.base}/health/readiness", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            pytest.fail(f"LiteLLM container exited during startup:\n{proxy.logs()}")
        time.sleep(2)
    pytest.fail(f"LiteLLM not ready after {STARTUP_TIMEOUT_S}s:\n{proxy.logs()}")


class Postgres:
    """A throwaway Postgres in its own container, so the proxy writes spend logs like production."""

    def __init__(self, name: str, port: int, log_path: Path) -> None:
        self.name = name
        self.port = port
        self.log_path = log_path
        self.host = "127.0.0.1"
        self._proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"postgresql://postgres@{self.host}:{self.port}/postgres"

    def start(self) -> None:
        # Same port both sides, like the proxy: host-network runners ignore the mapping.
        command = (
            "initdb -D /tmp/pgdata -U postgres -A trust -E UTF8 --locale=C.UTF-8 >/dev/null && "
            "echo 'host all all all trust' >> /tmp/pgdata/pg_hba.conf && "
            f"exec postgres -D /tmp/pgdata -p {self.port} -c listen_addresses=* "
            "-c unix_socket_directories=/tmp -c fsync=off"
        )
        with self.log_path.open("w") as log:
            self._proc = subprocess.Popen(
                [
                    *_runner(),
                    "--name",
                    self.name,
                    "-p",
                    f"127.0.0.1:{self.port}:{self.port}",
                    "--entrypoint",
                    "sh",
                    postgres_image(),
                    "-c",
                    command,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self._wait_ready()
        self.host = self._container_ip() or self.host

    def stop(self) -> None:
        subprocess.run([_cli(), "rm", "-f", self.name], capture_output=True, check=False)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _container_ip(self) -> str:
        # Bridge networks give the container an address the proxy container can reach; host networking leaves it empty.
        out = subprocess.run(
            [_cli(), "inspect", self.name, "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            probe = subprocess.run(
                [_cli(), "exec", self.name, "pg_isready", "-h", "127.0.0.1", "-p", str(self.port)],
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0:
                return
            if self._proc is not None and self._proc.poll() is not None:
                pytest.fail(f"Postgres container exited during startup:\n{self.log_path.read_text()}")
            time.sleep(1)
        pytest.fail(f"Postgres not ready after {STARTUP_TIMEOUT_S}s:\n{self.log_path.read_text()}")

    def query(self, sql: str) -> list[list[str]]:
        """Runs sql and returns its rows as lists of strings."""
        out = subprocess.run(
            [
                _cli(),
                "exec",
                self.name,
                "psql",
                "-h",
                "127.0.0.1",
                "-p",
                str(self.port),
                "-U",
                "postgres",
                "-X",
                "-A",
                "-t",
                "-F",
                "\t",
                "-c",
                sql,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        return [line.split("\t") for line in out.stdout.splitlines() if line]


class Proxy:
    def __init__(self, base: str, name: str, log_path: Path, database: Postgres) -> None:
        self.base = base
        self.name = name
        self.log_path = log_path
        self.database = database
        self.client = httpx.Client(base_url=base, timeout=60, headers={"authorization": f"Bearer {MASTER_KEY}"})

    def logs(self) -> str:
        return self.log_path.read_text()

    def python(self, code: str) -> str:
        """Runs code in the proxy container, with its PYTHONPATH and LiteLLM install."""
        out = subprocess.run(
            [_cli(), "exec", self.name, "python", "-c", code], capture_output=True, text=True, check=False
        )
        assert out.returncode == 0, out.stderr
        return out.stdout

    def start_fake_mcp(self) -> None:
        subprocess.run([_cli(), "exec", "-d", self.name, "python", "/it/fake_mcp.py"], check=True)
        deadline = time.monotonic() + 30
        probe = f"import socket; socket.create_connection(('127.0.0.1', {MCP_PORT}), 1)"
        while subprocess.run(
            [_cli(), "exec", self.name, "python", "-c", probe], capture_output=True, check=False
        ).returncode:
            assert time.monotonic() < deadline, "fake MCP server did not start"
            time.sleep(1)

    def upstream_received(self) -> list[dict]:
        return self._fake_received(8099)

    def scanner_received(self) -> list[dict]:
        return self._fake_received(SCANNER_PORT)

    def _fake_received(self, port: int) -> list[dict]:
        # The fakes only listen inside the container.
        return json.loads(
            self.python(
                "import urllib.request; "
                f"print(urllib.request.urlopen('http://127.0.0.1:{port}/_received').read().decode())"
            )
        )
