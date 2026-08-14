"""Build the paper's `data/` tree on a Modal Volume, and report what is in it.

    modal run bench/modal_stage.py::stage          # fetch the Zenodo bundle
    modal run bench/modal_stage.py::report         # inventory what is staged
    modal run bench/modal_stage.py::stage --source DMS_ProteinGym_substitutions.zip

A thin adapter around `bench.eval_data` -- all logic lives there, so the same
code path can rebuild the tree on a laptop. This file only defines the
container, declares the Volume, and moves the inventory back.

The Volume is the only Modal-side state this repo owns that is *not* rebuilt
from a definition on every run: images are declared in code and thrown away,
a Volume persists whatever was last written to it. That asymmetry is why the
source URLs and checksums live in `bench/eval_data.py` and why every ingest
appends to `_manifest.json` at the Volume root -- between them, a populated
Volume can be traced back to the code that populated it, or rebuilt if it is
lost.

Fetching runs *here* rather than on a developer machine because the 3.59 GB has
to end up in the Volume, and routing it through a laptop to get there is pure
waste. Not for speed: Zenodo serves this at around 2 MB/s no matter who asks,
so the container's link is no better than a laptop's.

Both long steps report progress on a timer to flushed stdout, which is the only
window into a container: a multi-gigabyte fetch that prints nothing for ten
minutes cannot be told apart from one that has hung.

Unpacking goes straight into the mounted Volume rather than into `/tmp` and
then copying across, which is the pattern Modal documents. The reason to
deviate: fetch-unpack-copy peaks at roughly zip + unpacked + copy on the
container disk, and extraction here is idempotent (members overwrite in place),
so an interrupted run is repaired by re-running rather than by a rollback. The
zip itself is deleted the moment it is unpacked. `_manifest.json` is written
last and so doubles as the completion marker -- `stage` skips a source already
recorded there unless `--force`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from bench.eval_data import (
    PROTEINGYM,
    ZENODO,
    Source,
    fetch,
    inventory,
    unpack,
    write_manifest,
)

REPO = Path(__file__).parent.parent

# Mounted at /root/data so that the notebooks' relative paths -- everything is
# read as `data/Figure5_CASP_ProteinGym/...` from the repo root -- resolve
# unchanged inside the container. Same code, same paths, locally and remotely.
DATA_ROOT = "/root/data"

VOLUME_NAME = "msa-pairformer-eval-data"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# No torch, no GPU: this is I/O. numpy is here because the PPI and CASP15
# ground truth are pickled arrays, and `inventory` has to be able to load them
# to report their keys rather than guessing from the opcode stream.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26,<2.0")
    .add_local_python_source("bench")
)

app = modal.App("msa-pairformer-eval-data", image=image)

SOURCES: dict[str, Source] = {ZENODO.name: ZENODO, **PROTEINGYM}


def _staged(root: Path) -> set[str]:
    """Source names already recorded as ingested in the Volume's manifest."""
    path = root / "_manifest.json"
    if not path.exists():
        return set()
    manifest = json.loads(path.read_text())
    return {
        entry["name"]
        for ingest in manifest.get("ingests", [])
        for entry in ingest.get("sources", [])
    }


# Two hours, not the 300s default: 2.96 GB down plus ~10 GB of extraction into
# a network filesystem is minutes of work at best, and a timeout mid-extract
# leaves a tree with no manifest -- recoverable, but only by doing it again.
# No `ephemeral_disk` request: the zip is the only thing on the container disk
# and it is deleted as soon as it is unpacked.
@app.function(volumes={DATA_ROOT: volume}, timeout=7200)
def ingest(names: list[str], git_sha: str | None, force: bool) -> dict[str, Any]:
    """Fetch, verify, unpack and record each named source. Returns the inventory."""
    root = Path(DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    done = set() if force else _staged(root)

    entries: list[dict[str, Any]] = []
    for name in names:
        source = SOURCES[name]
        if name in done:
            print(f"{name}: already in the manifest, skipping (--force to redo)", flush=True)
            continue

        print(f"{name}: fetching {source.url}", flush=True)
        archive, size, observed = fetch(source, Path("/tmp"))
        print(f"{name}: {size} bytes, md5 {observed}"
              f"{' (matches published)' if source.published else ' (observed only)'}",
              flush=True)

        members = unpack(archive, root, source.dest)
        archive.unlink()
        print(f"{name}: unpacked {members} members into {source.dest or '<root>'}", flush=True)

        entries.append(
            {
                "name": name,
                "url": source.url,
                "bytes": size,
                "md5": observed,
                "md5_published": source.md5,
                "dest": source.dest,
                "members": members,
            }
        )

    if entries:
        write_manifest(root, entries, git_sha)
        volume.commit()

    return inventory(root)


@app.function(volumes={DATA_ROOT: volume}, timeout=1800)
def survey() -> dict[str, Any]:
    """Inventory the Volume as it stands, changing nothing."""
    return inventory(Path(DATA_ROOT))


def _git_sha() -> str | None:
    from bench.provenance import git_info

    return git_info()["commit"]


def _save(result: dict[str, Any], out: str) -> None:
    path = Path(out or REPO / "bench" / "results" / "eval_data_inventory.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")


def _summarise(result: dict[str, Any]) -> None:
    total_gb = result["total_bytes"] / 1e9
    print(f"\n{result['total_files']} files, {total_gb:.2f} GB under {result['root']}")
    for name, entry in result["tree"].items():
        print(f"  {entry['files']:>6}  {entry['bytes'] / 1e6:>10.1f} MB  {name}")


@app.local_entrypoint()
def stage(source: str = "", force: bool = False, out: str = "") -> None:
    """Populate the Volume. Defaults to the Zenodo bundle alone.

    The bundle is fetched first and on its own deliberately: it may already
    carry the ProteinGym MSAs, in which case pulling `DMS_msa_files.zip` (1.5
    GB) would stage a second copy of data we already have. Inventory first,
    then name whichever ProteinGym archive is genuinely missing.
    """
    names = [source] if source else [ZENODO.name]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; known: {sorted(SOURCES)}")

    result = ingest.remote(names, _git_sha(), force)
    _summarise(result)
    _save(result, out)


@app.local_entrypoint()
def report(out: str = "") -> None:
    """Inventory the Volume without writing to it."""
    result = survey.remote()
    _summarise(result)
    _save(result, out)
