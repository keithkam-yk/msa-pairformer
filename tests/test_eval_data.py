"""The parts of the data staging path that can be checked without the network.

`bench/eval_data.py` is the logic behind `bench/modal_stage.py`, and all of it
runs here against a tmpdir and a fake server -- no network, no Modal.

The fetch tests matter most and are the reason `fetch` takes an `opener`. Its
job is to survive a source that ends a transfer early and *cleanly*, which is
what Zenodo does to this bundle: no exception is raised, so the failure is
invisible to everything except a checksum, and the recovery path appends to a
partial file, where an off-by-one offset produces a result of exactly the right
length and entirely corrupt. A real network exercises none of that on demand.

The rest are what a later ticket trusts: the inventory *is* the deliverable of
the staging ticket, and the manifest is what stops a re-run from re-pulling
three and a half gigabytes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import pytest

from bench.eval_data import Source, fetch, inventory, unpack, write_manifest


class FakeResponse:
    """One HTTP response that hands back a slice of `body`, possibly truncated.

    `cut` reproduces the failure this whole retry path exists for: the stream
    ends early and *cleanly*, with no exception, so a reader that trusts EOF
    walks away with a partial file.
    """

    def __init__(self, body: bytes, start: int, status: int, cut: int | None = None):
        self.status = status
        self._data = body[start : start + cut] if cut is not None else body[start:]
        self._pos = 0
        self.headers = {"Content-Length": str(len(self._data))}

    def read(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class FakeServer:
    """Serves `body`, cutting the stream short on the attempts named in `cuts`."""

    def __init__(self, body: bytes, cuts: dict[int, int], honour_range: bool = True):
        self.body = body
        self.cuts = cuts
        self.honour_range = honour_range
        self.requests: list[str | None] = []

    def __call__(self, request) -> FakeResponse:
        rng = request.get_header("Range")
        self.requests.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng and self.honour_range else 0
        status = 206 if rng and self.honour_range else 200
        return FakeResponse(self.body, start, status, self.cuts.get(len(self.requests)))


def _zip_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Figure2_x/a.txt", "hello" * 100)
    return buf.getvalue()


def _source(body: bytes, md5: str | None = None) -> Source:
    return Source(
        name="b.zip", url="https://example.invalid/b.zip", size=len(body),
        md5=md5, published=True, dest="", note="",
    )


def _tree(root: Path) -> None:
    (root / "Figure2_x").mkdir(parents=True)
    with (root / "Figure2_x" / "target_data.pkl").open("wb") as f:
        pickle.dump({"1A2B": {"seq": "ACD", "contacts": [[0, 1]]}}, f)
    (root / "Figure2_x" / "scores.csv").write_text("target,p_at_k\n1A2B,0.5\n2C3D,0.6\n")


def test_inventory_reports_pickle_shape(tmp_path: Path) -> None:
    """The pickle probe answers "how many targets, keyed how" without the schema."""
    _tree(tmp_path)
    inv = inventory(tmp_path)

    probe = inv["pickles"]["Figure2_x/target_data.pkl"]
    assert probe["type"] == "dict"
    assert probe["n_keys"] == 1
    assert probe["keys"] == ["1A2B"]
    assert probe["value_keys"] == ["seq", "contacts"]


def test_inventory_reads_small_csv_headers(tmp_path: Path) -> None:
    """Row counts exclude the header, so a stored results table is countable."""
    _tree(tmp_path)
    csv = inventory(tmp_path)["csvs"]["Figure2_x/scores.csv"]
    assert csv["header"] == "target,p_at_k"
    assert csv["rows"] == 2


def test_unreadable_pickle_is_a_finding_not_a_failure(tmp_path: Path) -> None:
    """A pickle needing classes we lack must still report what it references."""
    tmp_path.joinpath("bad.pkl").write_bytes(b"\x80\x04not-a-pickle")
    probe = inventory(tmp_path)["pickles"]["bad.pkl"]
    assert "load_error" in probe


def test_tree_counts_against_the_directory_at_the_depth_limit(tmp_path: Path) -> None:
    """Sibling directories below the limit must stay distinct, not merge upward."""
    for sub in ("uniref_msas", "logan_msas"):
        deep = tmp_path / "Figure5" / "ProteinGym" / sub / "extra"
        deep.mkdir(parents=True)
        (deep / "x.a3m").write_text("seq")

    tree = inventory(tmp_path, tree_depth=3)["tree"]
    assert "Figure5/ProteinGym/uniref_msas" in tree
    assert "Figure5/ProteinGym/logan_msas" in tree
    assert "Figure5/ProteinGym" not in tree


def test_unpack_is_idempotent(tmp_path: Path) -> None:
    """Re-running a partially finished ingest repairs it rather than doubling it."""
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Figure2_x/a.txt", "hello")

    root = tmp_path / "data"
    assert unpack(archive, root, "") == 1
    assert unpack(archive, root, "") == 1
    assert inventory(root)["total_files"] == 1


def test_fetch_resumes_after_a_silent_truncation(tmp_path: Path) -> None:
    """The exact Zenodo failure: a clean early EOF must not pass as a download."""
    body = _zip_bytes()
    server = FakeServer(body, cuts={1: len(body) // 3})

    path, size, md5 = fetch(_source(body), tmp_path, opener=server, backoff=0)

    assert size == len(body)
    assert path.read_bytes() == body
    assert md5 == hashlib.md5(body).hexdigest()
    # Second request resumed from where the first stopped, rather than restarting.
    assert server.requests == [None, f"bytes={len(body) // 3}-"]


def test_fetch_restarts_when_the_server_ignores_range(tmp_path: Path) -> None:
    """Appending a full body to a partial file yields the right length and garbage."""
    body = _zip_bytes()
    server = FakeServer(body, cuts={1: len(body) // 3}, honour_range=False)

    path, size, md5 = fetch(_source(body), tmp_path, opener=server, backoff=0)

    assert size == len(body)
    assert path.read_bytes() == body
    assert md5 == hashlib.md5(body).hexdigest()


def test_fetch_gives_up_rather_than_returning_a_short_file(tmp_path: Path) -> None:
    """A source that will not stay up must fail loudly, not quietly deliver less."""
    body = _zip_bytes()
    server = FakeServer(body, cuts=dict.fromkeys(range(1, 5), 10))

    with pytest.raises(ValueError, match="short after 3 attempts"):
        fetch(_source(body), tmp_path, attempts=3, opener=server, backoff=0)


def test_fetch_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    """A published checksum is enforced; the wrong bytes never reach the tree."""
    body = _zip_bytes()

    with pytest.raises(ValueError, match="expected md5"):
        fetch(_source(body, md5="0" * 32), tmp_path, opener=FakeServer(body, {}), backoff=0)


def test_fetch_rejects_a_non_zip(tmp_path: Path) -> None:
    """An error page served with a 200 is not a silent success."""
    body = b"<html>not here</html>" * 50

    with pytest.raises(ValueError, match="not a zip"):
        fetch(_source(body), tmp_path, opener=FakeServer(body, {}), backoff=0)


def test_manifest_accumulates_ingests(tmp_path: Path) -> None:
    """Each ingest appends, so the tree records its whole provenance, not the last."""
    write_manifest(tmp_path, [{"name": "one"}], "sha1")
    write_manifest(tmp_path, [{"name": "two"}], "sha2")

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert [i["git_sha"] for i in manifest["ingests"]] == ["sha1", "sha2"]
    assert [i["sources"][0]["name"] for i in manifest["ingests"]] == ["one", "two"]
