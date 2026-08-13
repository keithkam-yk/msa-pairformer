"""The parts of the data staging path that can be checked without the network.

`bench/eval_data.py` is the logic behind `bench/modal_stage.py`; the fetch is
untestable here by definition, but everything downstream of it -- unpacking,
the inventory, and the manifest that marks an ingest complete -- runs on a
hand-built tree in a tmpdir. These are the pieces a later ticket trusts: the
inventory *is* the deliverable of the staging ticket, and the manifest is what
stops a re-run from re-pulling three gigabytes.
"""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

from bench.eval_data import inventory, unpack, write_manifest


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


def test_unpack_is_idempotent(tmp_path: Path) -> None:
    """Re-running a partially finished ingest repairs it rather than doubling it."""
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Figure2_x/a.txt", "hello")

    root = tmp_path / "data"
    assert unpack(archive, root, "") == 1
    assert unpack(archive, root, "") == 1
    assert inventory(root)["total_files"] == 1


def test_manifest_accumulates_ingests(tmp_path: Path) -> None:
    """Each ingest appends, so the tree records its whole provenance, not the last."""
    write_manifest(tmp_path, [{"name": "one"}], "sha1")
    write_manifest(tmp_path, [{"name": "two"}], "sha2")

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert [i["git_sha"] for i in manifest["ingests"]] == ["sha1", "sha2"]
    assert [i["sources"][0]["name"] for i in manifest["ingests"]] == ["one", "two"]
