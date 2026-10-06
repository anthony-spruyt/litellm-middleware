from pathlib import Path

from conftest import CALLBACK, MOUNT_PATH, REPO_ROOT

PACKAGE_SRC = REPO_ROOT / "src" / "litellm_middleware"


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_image_ships_only_the_package_sources(package_files):
    expected = {f"litellm_middleware/{p.relative_to(PACKAGE_SRC)}" for p in PACKAGE_SRC.rglob("*.py")}

    assert _files(package_files) == expected


def test_proxy_imports_the_callback_from_the_mount(proxy):
    module = CALLBACK.rsplit(".", 1)[0]
    out = proxy.python(f"import {module} as m; print(m.__file__)")

    assert out.strip() == f"{MOUNT_PATH}/{module.replace('.', '/')}.py"
