"""Fails a LiteLLM bump PR while the middleware on the base branch differs from the latest published release.

spruyt-labs deploys the LiteLLM pin from main next to the released middleware image, so a bump must not merge
until every image change it was tested with has shipped.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PIN_FILE = "litellm-image.yaml"
IMAGE_PATHS = ("src/litellm_middleware", "Dockerfile", ".dockerignore")
GATE_PATHS = (
    "scripts/release_gate.py",
    ".github/workflows/release-gate.yaml",
    "tests/unit/scripts/test_release_gate.py",
    "tests/integration/conftest.py",
)
MIDDLEWARE_REPOSITORY = "ghcr.io/anthony-spruyt/litellm-middleware"
SPRUYT_LABS_VALUES_URL = (
    "https://raw.githubusercontent.com/anthony-spruyt/spruyt-labs/main/cluster/apps/litellm/litellm/app/values.yaml"
)
DEPLOYED_PIN = re.compile(
    rf"^\s*repository:\s*{re.escape(MIDDLEWARE_REPOSITORY)}\s*\n\s*tag:\s*\"?([^\s\"@]+@sha256:[a-f0-9]{{64}})\"?\s*$",
    re.MULTILINE,
)
RELEASE_FIRST = (
    "release the middleware first: merge the release-please PR, "
    "wait for the release to be published (image pushed), then re-run this check"
)


@dataclass(frozen=True)
class Result:
    ok: bool
    message: str


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def changed(repo: Path, *revs: str, paths: tuple[str, ...] = ()) -> list[str]:
    return git(repo, "diff", "--name-only", "--no-renames", *revs, "--", *paths).split()


def latest_release_tag() -> str | None:
    proc = subprocess.run(
        ["gh", "release", "view", "--json", "tagName", "--jq", ".tagName"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.strip() or None
    if "release not found" in proc.stderr:
        return None
    raise RuntimeError(f"gh release view failed: {proc.stderr.strip()}")


def listing(paths: list[str]) -> str:
    return "\n".join(f"  {path}" for path in paths)


def evaluate(repo: Path, base: str, head: str, release_tag: Callable[[], str | None]) -> Result:
    if PIN_FILE not in changed(repo, f"{base}...{head}"):
        return Result(True, f"not a LiteLLM bump ({PIN_FILE} unchanged)")

    gate = changed(repo, f"{base}...{head}", paths=GATE_PATHS)
    if gate:
        return Result(
            False,
            "a LiteLLM bump must not change the release gate; drop these changes. "
            f"A gate change needs its own PR, reviewed by the code owner:\n{listing(gate)}",
        )

    in_pr = changed(repo, f"{base}...{head}", paths=IMAGE_PATHS)
    if in_pr:
        return Result(
            False,
            "this LiteLLM bump also changes the middleware image. Land the change in its own PR, "
            f"then {RELEASE_FIRST}:\n{listing(in_pr)}",
        )

    tag = release_tag()
    if tag is None:
        return Result(False, f"no published release of the middleware; {RELEASE_FIRST}")

    unreleased = changed(repo, tag, base, paths=IMAGE_PATHS)
    if unreleased:
        return Result(
            False, f"{base} changes the middleware image since {tag}; {RELEASE_FIRST}:\n{listing(unreleased)}"
        )

    return Result(True, f"middleware image on {base} matches release {tag}")


def deployed_image(values: str) -> str:
    """Returns the digest-pinned middleware image that spruyt-labs mounts next to LiteLLM."""
    pins = DEPLOYED_PIN.findall(values)
    if len(pins) != 1:
        raise ValueError(f"expected one digest-pinned {MIDDLEWARE_REPOSITORY} image in the values, found {len(pins)}")
    return f"{MIDDLEWARE_REPOSITORY}:{pins[0]}"


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", help="base branch ref, e.g. origin/main")
    parser.add_argument("--head", help="PR head ref or sha")
    parser.add_argument("--release-tag", help="skip the GitHub lookup and use this tag")
    parser.add_argument(
        "--deployed-image", action="store_true", help="print the middleware image spruyt-labs deploys, then exit"
    )
    args = parser.parse_args(argv)

    if args.deployed_image:
        print(deployed_image(fetch(SPRUYT_LABS_VALUES_URL)))
        return 0
    if not (args.base and args.head):
        parser.error("--base and --head are required")

    lookup = (lambda: args.release_tag) if args.release_tag else latest_release_tag
    result = evaluate(args.repo, args.base, args.head, lookup)
    if result.ok:
        print(f"::notice::{result.message}")
        return 0
    first, _, rest = result.message.partition("\n")
    print(f"::error::{first}")
    if rest:
        print(rest)
    return 1


if __name__ == "__main__":
    sys.exit(main())
