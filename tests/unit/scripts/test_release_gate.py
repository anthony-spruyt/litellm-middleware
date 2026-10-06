import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "release_gate.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("release_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_gate"] = module
    spec.loader.exec_module(module)
    return module


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.git("init", "-q", "-b", "main")
        self.write("src/litellm_middleware/pipeline.py", "v1\n")
        self.write("litellm-image.yaml", "image: litellm:v1\n")
        self.write("README.md", "readme\n")
        self.commit("initial")
        self.git("tag", "v1.0.0")

    def git(self, *args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                *args,
            ],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self, name: str, content: str) -> None:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def branch(self, name: str) -> None:
        self.git("checkout", "-q", "-b", name)

    def checkout(self, name: str) -> None:
        self.git("checkout", "-q", name)


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path)


def check(gate, repo, tag="v1.0.0"):
    return gate.evaluate(repo.path, base="main", head="bump", release_tag=lambda: tag)


def test_non_bump_pr_passes_without_looking_up_a_release(gate, repo):
    repo.branch("bump")
    repo.write("README.md", "changed\n")
    repo.commit("docs")

    def no_lookup():
        raise AssertionError("release lookup must not run for non-bump PRs")

    result = gate.evaluate(repo.path, base="main", head="bump", release_tag=no_lookup)

    assert result.ok
    assert "not a LiteLLM bump" in result.message


def test_bump_passes_when_image_contents_match_the_release(gate, repo):
    repo.write("README.md", "unreleased docs\n")
    repo.commit("docs on main after the release")
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.commit("bump")

    result = check(gate, repo)

    assert result.ok
    assert "v1.0.0" in result.message


@pytest.mark.parametrize(
    "path", ["src/litellm_middleware/pipeline.py", "src/litellm_middleware/new/new.py", "Dockerfile", ".dockerignore"]
)
def test_bump_fails_when_main_has_unreleased_image_changes(gate, repo, path):
    repo.write(path, "unreleased fix\n")
    repo.commit("fix on main, not released")
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.commit("bump")

    result = check(gate, repo)

    assert not result.ok
    assert path in result.message
    assert "release the middleware first" in result.message
    assert "merge the release-please PR, wait for the release to be published (image pushed), then re-run" in (
        result.message
    )


def test_bump_fails_against_live_base_even_when_branched_before_the_fix(gate, repo):
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.commit("bump")
    repo.checkout("main")
    repo.write("src/litellm_middleware/pipeline.py", "unreleased fix\n")
    repo.commit("fix lands after the bump PR opened")

    result = check(gate, repo)

    assert not result.ok


def test_bump_passes_on_rerun_once_the_fix_is_released(gate, repo):
    repo.write("src/litellm_middleware/pipeline.py", "fix\n")
    repo.commit("fix")
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.commit("bump")
    assert not check(gate, repo).ok

    repo.checkout("main")
    repo.write("CHANGELOG.md", "1.0.1\n")
    repo.commit("chore(main): release 1.0.1")
    repo.git("tag", "v1.0.1")

    assert check(gate, repo, tag="v1.0.1").ok


@pytest.mark.parametrize("path", ["src/litellm_middleware/pipeline.py", "Dockerfile"])
def test_bump_fails_when_the_pr_itself_changes_the_image(gate, repo, path):
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.write(path, "compat fix in the bump PR\n")
    repo.commit("bump with fix")

    result = check(gate, repo)

    assert not result.ok
    assert path in result.message
    assert "own PR" in result.message


@pytest.mark.parametrize(
    "path",
    ["scripts/release_gate.py", ".github/workflows/release-gate.yaml", "tests/unit/scripts/test_release_gate.py"],
)
def test_bump_fails_when_the_pr_also_changes_the_gate(gate, repo, path):
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.write(path, "always pass\n")
    repo.commit("bump and loosen the gate")

    result = check(gate, repo)

    assert not result.ok
    assert path in result.message
    assert "reviewed by the code owner" in result.message


@pytest.mark.parametrize(
    "move_pin",
    [["mv", "litellm-image.yaml", "litellm.yaml"], ["rm", "-q", "litellm-image.yaml"]],
    ids=["rename", "delete"],
)
def test_moving_the_pin_counts_as_a_bump(gate, repo, move_pin):
    repo.write("src/litellm_middleware/pipeline.py", "unreleased fix\n")
    repo.commit("fix on main, not released")
    repo.branch("bump")
    repo.git(*move_pin)
    repo.commit("move the pin")

    result = check(gate, repo)

    assert not result.ok
    assert "release the middleware first" in result.message


def test_bump_fails_when_nothing_is_released(gate, repo):
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.commit("bump")

    result = check(gate, repo, tag=None)

    assert not result.ok
    assert "no published release" in result.message


def test_image_paths_cover_the_dockerfile_and_its_copy_sources(gate):
    sources = re.findall(r"^(?:COPY|ADD)\s+(?:--\S+\s+)*(\S+)", (REPO_ROOT / "Dockerfile").read_text(), re.MULTILINE)

    assert {source.rstrip("/") for source in sources} | {"Dockerfile", ".dockerignore"} <= set(gate.IMAGE_PATHS)


def fake_gh(tmp_path, monkeypatch, script: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!/usr/bin/env bash\n{script}\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_latest_release_tag_returns_the_tag_gh_reports(gate, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, 'echo "v1.2.3"')

    assert gate.latest_release_tag() == "v1.2.3"


def test_latest_release_tag_is_none_without_a_release(gate, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, 'echo "release not found" >&2; exit 1')

    assert gate.latest_release_tag() is None


def test_latest_release_tag_raises_on_other_gh_errors(gate, tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, 'echo "HTTP 401: Bad credentials" >&2; exit 1')

    with pytest.raises(RuntimeError, match="Bad credentials"):
        gate.latest_release_tag()


def test_main_exits_nonzero_and_reports_the_failure(gate, repo, capsys):
    repo.write("src/litellm_middleware/pipeline.py", "unreleased fix\n")
    repo.commit("fix")
    repo.branch("bump")
    repo.write("litellm-image.yaml", "image: litellm:v2\n")
    repo.commit("bump")

    code = gate.main(["--repo", str(repo.path), "--base", "main", "--head", "bump", "--release-tag", "v1.0.0"])

    assert code == 1
    assert "::error::" in capsys.readouterr().out


def test_main_exits_zero_for_a_non_bump(gate, repo, capsys):
    repo.branch("bump")
    repo.write("README.md", "changed\n")
    repo.commit("docs")

    code = gate.main(["--repo", str(repo.path), "--base", "main", "--head", "bump"])

    assert code == 0
    assert "::notice::" in capsys.readouterr().out
