import json

import numpy as np

from lebse.eval_pairs import build_parser, deranged, load_pairs


def test_deranged_has_no_fixed_point():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 50, 500):
        p = deranged(n, rng)
        assert sorted(p.tolist()) == list(range(n))  # a permutation
        assert not np.any(p == np.arange(n))  # no i -> i (every negative is a cross-pairing)


def test_load_pairs(tmp_path):
    f = tmp_path / "pairs.jsonl"
    f.write_text(
        json.dumps({"a": "opinion one body", "b": "opinion two body"})
        + "\n"
        + "\n"  # blank line ignored
        + json.dumps({"a": "x", "b": ""})
        + "\n"  # empty b dropped
        + json.dumps({"a": "third a", "b": "third b"})
        + "\n",
        encoding="utf-8",
    )
    a, b = load_pairs(str(f))
    assert a == ["opinion one body", "third a"]
    assert b == ["opinion two body", "third b"]


def test_load_pairs_limit(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text("".join(json.dumps({"a": str(i), "b": str(i)}) + "\n" for i in range(10)))
    a, b = load_pairs(str(f), limit=3)
    assert len(a) == 3 and len(b) == 3


def test_cli_prog():
    assert build_parser().prog == "lebse-eval-pairs"
