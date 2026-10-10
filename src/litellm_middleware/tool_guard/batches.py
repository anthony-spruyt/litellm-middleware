"""Packs scanner request items into JSON bodies that stay under a byte limit."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

# Same encoding httpx uses for json=..., so a body's length is the length on the wire.
_JSON_OPTIONS = {"ensure_ascii": False, "separators": (",", ":"), "allow_nan": False}
_EMPTY_BODY_BYTES = len(b'{"new":[],"known":[]}')


@dataclass
class Batch:
    new: list[bytes] = field(default_factory=list)
    known: list[bytes] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)
    size: int = _EMPTY_BODY_BYTES

    @property
    def body(self) -> bytes:
        return b'{"new":[' + b",".join(self.new) + b'],"known":[' + b",".join(self.known) + b"]}"

    def growth(self, fragment: bytes, is_new: bool) -> int:
        siblings = self.new if is_new else self.known
        return len(fragment) + (1 if siblings else 0)

    def add(self, digest: str, fragment: bytes, is_new: bool) -> None:
        self.size += self.growth(fragment, is_new)
        (self.new if is_new else self.known).append(fragment)
        self.hashes.append(digest)


def _encode(value: object) -> bytes:
    return json.dumps(value, **_JSON_OPTIONS).encode("utf-8")


def _fragments(new: Iterable[tuple[str, str]], known: Iterable[str]) -> Iterator[tuple[str, bytes, bool]]:
    for digest, text in new:
        yield digest, _encode({"hash": digest, "text": text}), True
    for digest in known:
        yield digest, _encode(digest), False


def batches(new: Iterable[tuple[str, str]], known: Iterable[str], limit: int) -> Iterator[Batch]:
    """Yields request bodies of at most limit bytes, in the order given, new items before known hashes.

    An item that cannot fit in a body on its own is left out. Items are encoded as the caller consumes batches.
    """
    batch = Batch()
    for digest, fragment, is_new in _fragments(new, known):
        if _EMPTY_BODY_BYTES + len(fragment) > limit:
            continue
        if batch.size + batch.growth(fragment, is_new) > limit:
            yield batch
            batch = Batch()
        batch.add(digest, fragment, is_new)
    if batch.hashes:
        yield batch
