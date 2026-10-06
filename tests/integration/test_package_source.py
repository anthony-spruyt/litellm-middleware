from conftest import PACKAGE_IMAGE_ENV, REPO_ROOT, package_build_context

RELEASED = "ghcr.io/anthony-spruyt/litellm-middleware:1.0.0@sha256:" + "ab" * 32


def test_package_is_built_from_the_checkout_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(PACKAGE_IMAGE_ENV, raising=False)

    assert package_build_context(tmp_path) == REPO_ROOT


def test_package_is_taken_from_a_released_image_when_one_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv(PACKAGE_IMAGE_ENV, RELEASED)

    context = package_build_context(tmp_path)

    assert [p.name for p in context.iterdir()] == ["Dockerfile"]
    assert (context / "Dockerfile").read_text() == f"FROM {RELEASED}\n"
