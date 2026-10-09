"""Boots the pinned LiteLLM image with the package image's files mounted where the pod mounts them."""

from __future__ import annotations

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
STARTUP_TIMEOUT_S = 240
MCP_SERVER = "itmcp"
MCP_PORT = 8098


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
                }
            ],
            "litellm_settings": {"callbacks": [CALLBACK]},
            "general_settings": {"include_call_id_in_error_body": True, "master_key": MASTER_KEY},
            # Down at startup so the proxy's tool mapping stays cold, like a DB-loaded server after a restart.
            "mcp_servers": {MCP_SERVER: {"url": f"http://127.0.0.1:{MCP_PORT}/mcp", "transport": "http"}},
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
    for fake in ("fake_upstream.py", "fake_mcp.py"):
        (work / fake).write_bytes((HERE / fake).read_bytes())
    _world_readable(work)

    name = f"litellm-it-{uuid.uuid4().hex[:8]}"
    port = _free_port()
    log_path = work / "proxy.log"
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
                "-v",
                f"{package_files}:{MOUNT_PATH}:ro",
                "-v",
                f"{work / 'config.yaml'}:/app/config.yaml:ro",
                "-v",
                f"{work / 'fake_upstream.py'}:/it/fake_upstream.py:ro",
                "-v",
                f"{work / 'fake_mcp.py'}:/it/fake_mcp.py:ro",
                "--entrypoint",
                "sh",
                litellm_image(),
                "-c",
                f"python /it/fake_upstream.py & exec litellm --config /app/config.yaml --port {port}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    proxy = Proxy(f"http://127.0.0.1:{port}", name, log_path)
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


class Proxy:
    def __init__(self, base: str, name: str, log_path: Path) -> None:
        self.base = base
        self.name = name
        self.log_path = log_path
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
        # The fake upstream only listens inside the container.
        return yaml.safe_load(
            self.python(
                "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8099/_received').read().decode())"
            )
        )
