import json

import httpx

from litellm_middleware.tool_guard.batches import batches


def _id(name):
    return f"sha256:{name}"


def _bodies(new, known, limit):
    return [b.body for b in batches(new, known, limit)]


def _httpx_body(new, known):
    payload = {"new": [{"hash": h, "text": t} for h, t in new], "known": known}
    return httpx.Request("POST", "http://scanner.test", json=payload).content


def _sizes(new, known):
    return len(_httpx_body(new, known))


def test_everything_that_fits_goes_in_one_batch_encoded_like_the_http_client():
    new = [(_id("a"), 'say "hi"\n'), (_id("b"), "\u00e9 \U0001f600")]
    known = [_id("c"), _id("d")]

    [batch] = list(batches(new, known, 1_000_000))

    assert batch.body == _httpx_body(new, known)
    assert batch.hashes == [_id("a"), _id("b"), _id("c"), _id("d")]


def test_nothing_to_send_makes_no_batch():
    assert list(batches([], [], 1000)) == []


def test_batch_body_may_be_exactly_the_limit():
    new = [(_id("a"), "x" * 50), (_id("b"), "y" * 50)]
    exact = _sizes(new, [])

    assert len(_bodies(new, [], exact)) == 1
    assert len(_bodies(new, [], exact - 1)) == 2


def test_items_keep_their_order_across_batches():
    new = [(f"sha256:{i}", "x" * 40) for i in range(6)]
    per_two = _sizes(new[:2], [])

    out = list(batches(new, [], per_two))

    assert [b.hashes for b in out] == [[_id("0"), _id("1")], [_id("2"), _id("3")], [_id("4"), _id("5")]]
    assert all(len(b.body) <= per_two for b in out)


def test_limit_counts_encoded_bytes_so_escapes_batch_smaller_than_plain_text():
    limit = 500
    kinds = {"plain": "a", "quote": '"', "control": "\x01"}
    items = {kind: [(f"sha256:{kind}{i}", char * 60) for i in range(4)] for kind, char in kinds.items()}

    counts = {kind: [len(b.hashes) for b in batches(group, [], limit)] for kind, group in items.items()}

    assert counts == {"plain": [4], "quote": [3, 1], "control": [1, 1, 1, 1]}
    for group in items.values():
        for batch in batches(group, [], limit):
            assert len(batch.body) <= limit
            assert len(json.loads(batch.body)["new"]) == len(batch.hashes)


def test_limit_counts_utf8_bytes_not_characters():
    wide = [(_id("a"), "é" * 50), (_id("b"), "é" * 50)]
    limit = _sizes(wide[:1], [])

    assert [b.hashes for b in batches(wide, [], limit)] == [[_id("a")], [_id("b")]]


def test_item_over_the_limit_on_its_own_is_dropped_and_the_rest_still_sent():
    new = [(_id("a"), "x" * 20), (_id("big"), "y" * 500), (_id("b"), "x" * 20)]
    limit = _sizes(new[:1], [])

    out = list(batches(new, [], limit))

    assert [b.hashes for b in out] == [[_id("a")], [_id("b")]]


def test_item_over_the_limit_is_dropped_when_first_or_last():
    new = [(_id("big"), "y" * 500), (_id("a"), "x" * 20), (_id("big2"), "z" * 500)]
    limit = _sizes(new[1:2], [])

    assert [b.hashes for b in batches(new, [], limit)] == [[_id("a")]]


def test_known_hashes_follow_new_items_and_fill_the_same_batch():
    new = [(_id("a"), "x" * 40)]
    known = [_id("k1"), _id("k2")]

    [batch] = list(batches(new, known, _sizes(new, known)))

    assert json.loads(batch.body) == {"new": [{"hash": _id("a"), "text": "x" * 40}], "known": known}


def test_known_hashes_respect_the_limit():
    known = [f"sha256:{i:064d}" for i in range(10)]
    limit = _sizes([], known[:4])

    out = list(batches([], known, limit))

    assert [len(b.hashes) for b in out] == [4, 4, 2]
    assert all(len(b.body) <= limit for b in out)
    assert [h for b in out for h in b.hashes] == known


def test_a_batch_with_new_items_spills_known_hashes_into_the_next_batch():
    new = [(_id("a"), "x" * 40)]
    known = [_id("k1"), _id("k2")]
    limit = _sizes(new, known[:1])

    out = list(batches(new, known, limit))

    assert [b.hashes for b in out] == [[_id("a"), _id("k1")], [_id("k2")]]
    assert json.loads(out[1].body) == {"new": [], "known": [_id("k2")]}


def test_batches_are_encoded_only_as_they_are_consumed():
    seen = []

    def new():
        for i in range(3):
            seen.append(i)
            yield f"sha256:{i}", "x" * 40

    stream = batches(new(), [], _sizes([(_id("0"), "x" * 40)], []))

    next(stream)

    assert seen == [0, 1]
