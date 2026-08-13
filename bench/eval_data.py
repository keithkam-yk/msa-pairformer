"""Where the paper's published evaluation data comes from, and what is in it.

The analysis notebooks read everything from a repo-root `data/` tree by
relative path (`../../data/Figure5_CASP_ProteinGym/...`). None of that tree is
in git -- it is published separately, so the tree has to be rebuilt from its
sources before any benchmark can run. This module holds those sources and the
code that turns them into the tree, with no Modal and no torch in it, so the
same path runs on a laptop and in a container.

`bench/modal_stage.py` is the Modal adapter: it owns the Volume and calls in
here.

Checksums come in two strengths and the distinction matters. Zenodo publishes
an MD5, so `Source.md5` for the bundle attests that what we fetched is what the
authors deposited. ProteinGym publishes nothing, so its checksums are
*observed* -- recorded on first ingest, they detect later drift but say nothing
about whether the first ingest was the intended file. `Source.published` marks
which is which.
"""

from __future__ import annotations

import hashlib
import json
import pickletools
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

CHUNK = 1 << 22  # 4 MiB; large enough that a 3 GB fetch is not syscall-bound.


@dataclass(frozen=True)
class Source:
    """One downloadable archive and where its contents belong.

    `dest` is relative to the data root, so an archive whose members already
    carry the directory they belong in uses `dest=""` and unpacks at the root.
    `published` says whether `md5` came from the publisher or from us.
    """

    name: str
    url: str
    size: int | None
    md5: str | None
    published: bool
    dest: str
    note: str


# The paper's own bundle: Zenodo 10.5281/zenodo.21495175, CC-BY-4.0. Its
# entries are already `Figure2_.../`, `Figure5_.../`, so it unpacks at the root
# of the data tree and reproduces exactly the paths the notebooks expect.
#
# This is the *second* version of the deposit, and the version matters. The
# first (10.5281/zenodo.20603231, 2026-06-09, 2,959,081,472 bytes,
# md5 c205477a9cb93ad09bbc49ff6336b435) is a **truncated upload**: it downloads
# to its full declared length and matches its published MD5, so nothing about
# the transfer looks wrong, but the bytes themselves end mid-stream. Its last
# megabyte contains no end-of-central-directory record, no Zip64 EOCD and no
# central-directory entries at all, so no zip tool can open it. The 2026-07-22
# version below is 631 MB larger and ends in a well-formed EOCD.
#
# Pin the version DOI, never the concept DOI (10.5281/zenodo.20603230): the
# concept resolves to "latest", which would silently change the data under a
# benchmark whose whole purpose is comparability against fixed numbers.
ZENODO = Source(
    name="MSA_Pairformer_data.zip",
    url="https://zenodo.org/api/records/21495175/files/MSA_Pairformer_data.zip/content",
    size=3590716422,
    md5="0672b426338d9caeb9c2744d9bbc3e6e",
    published=True,
    dest="",
    note="MSA Pairformer analysis data (Zenodo 10.5281/zenodo.21495175, v2)",
)

# ProteinGym v1.3. Only fetched for pieces the Zenodo bundle turns out not to
# carry -- see the inventory on the staging ticket. Sizes are the host's
# Content-Length, and every one of these was confirmed to start with the zip
# magic rather than an HTML error page. Checksums stay None until a first
# ingest records one, and are observed rather than published either way.
PROTEINGYM_BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"

PROTEINGYM = {
    s.name: s
    for s in (
        Source(
            name="DMS_ProteinGym_substitutions.zip",
            url=f"{PROTEINGYM_BASE}/DMS_ProteinGym_substitutions.zip",
            size=43021128,
            md5=None,
            published=False,
            dest="Figure5_CASP_ProteinGym/ProteinGym",
            note="per-assay DMS csvs",
        ),
        Source(
            name="DMS_msa_files.zip",
            url=f"{PROTEINGYM_BASE}/DMS_msa_files.zip",
            size=1504361305,
            md5=None,
            published=False,
            dest="Figure5_CASP_ProteinGym/ProteinGym",
            note="reference MSAs; superseded if the bundle ships uniref_proteingym_msas/",
        ),
        Source(
            name="DMS_msa_weights.zip",
            url=f"{PROTEINGYM_BASE}/DMS_msa_weights.zip",
            size=45228459,
            md5=None,
            published=False,
            dest="Figure5_CASP_ProteinGym/ProteinGym",
            note="sequence reweighting files matching DMS_msa_files",
        ),
    )
}


class Progress:
    """Periodic one-line progress for work measured in bytes or in items.

    Ticks on elapsed time rather than on count, so a fast link does not flood
    the log and a slow one still reports. That matters more than usual here:
    this runs in a Modal container whose only window is streamed stdout, and a
    multi-gigabyte fetch that prints nothing for ten minutes is
    indistinguishable from one that has hung. Every line is flushed for the
    same reason -- buffered output arrives only at exit, which is exactly when
    it is no longer useful.
    """

    def __init__(self, label: str, total: int | None, unit: str = "B", every: float = 10.0):
        self.label = label
        self.total = total
        self.unit = unit
        self.every = every
        self.done = 0
        self.start = time.monotonic()
        self.last = self.start

    def _scale(self, n: float) -> str:
        return f"{n / 1e9:.2f} GB" if self.unit == "B" else f"{n:,.0f}"

    def advance(self, n: int = 1) -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            self.report()

    def report(self) -> None:
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed else 0.0
        line = f"  {self.label}: {self._scale(self.done)}"
        if self.total:
            pct = 100 * self.done / self.total
            remaining = (self.total - self.done) / rate if rate else 0.0
            line += f" / {self._scale(self.total)} ({pct:.0f}%), {remaining / 60:.1f} min left"
        # MB/s, not GB/s: the throttled sources this pulls from run at single
        # digit MB/s, and a rate column that reads 0.00 for the whole transfer
        # tells you nothing about whether it is moving.
        if self.unit == "B":
            print(f"{line}, {rate / 1e6:.1f} MB/s", flush=True)
        else:
            print(f"{line}, {rate:.1f} {self.unit}/s", flush=True)


def fetch(source: Source, into: Path) -> tuple[Path, int, str]:
    """Download to `into`, returning the path, observed size and observed MD5.

    Streams and hashes in one pass -- a 3 GB archive is not read twice. Raises
    if the source publishes a checksum and the download does not match it; an
    unpublished checksum is returned for recording, never enforced here.

    The zip check after the checksum is not redundant with it. A deposit can be
    served at its full declared length, match its published MD5, and still be a
    truncated archive -- that is exactly what the first version of the Zenodo
    bundle is. A matching checksum attests only that we received what was
    uploaded, never that what was uploaded is intact.
    """
    into.mkdir(parents=True, exist_ok=True)
    path = into / source.name
    digest = hashlib.md5()
    size = 0
    progress = Progress(f"fetch {source.name}", source.size)

    with urlopen(source.url) as response, path.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            progress.advance(len(chunk))
    progress.report()

    observed = digest.hexdigest()
    if source.md5 and observed != source.md5:
        raise ValueError(
            f"{source.name}: expected md5 {source.md5}, got {observed} ({size} bytes)"
        )
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{source.name}: not a zip -- the host likely served an error page")
    return path, size, observed


def unpack(archive: Path, root: Path, dest: str) -> int:
    """Extract into `root / dest`, returning the number of members written.

    Extraction is idempotent: members overwrite in place, so a run interrupted
    part-way is repaired by re-running rather than by clearing the tree.
    """
    target = root / dest if dest else root
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        # Member by member rather than `extractall`, which offers no hook to
        # report from. Extraction here writes into a network filesystem and is
        # the slowest step by a wide margin, so it is the one that most needs
        # to say how far along it is.
        progress = Progress(f"unpack {archive.name}", len(members), unit="files")
        for info in members:
            zf.extract(info, target)
            progress.advance()
        progress.report()
    return len(members)


def _probe_pickle(path: Path) -> dict[str, Any]:
    """Describe a pickle without needing the classes inside it.

    Loads if the container is plain (dict of arrays); falls back to reading the
    opcode stream for the class names it references, which is enough to say
    what a benchmark would have to import. Never raises -- a pickle we cannot
    read is a finding, not a failure.
    """
    out: dict[str, Any] = {"bytes": path.stat().st_size}
    try:
        import pickle

        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as exc:  # noqa: BLE001 -- the exception *is* the finding
        out["load_error"] = f"{type(exc).__name__}: {exc}"
        try:
            with path.open("rb") as f:
                ops = [
                    str(arg)
                    for op, arg, _ in pickletools.genops(f)
                    if op.name in ("GLOBAL", "STACK_GLOBAL", "SHORT_BINUNICODE")
                ]
            out["opcode_strings"] = sorted(set(ops))[:40]
        except Exception as inner:  # noqa: BLE001
            out["disassemble_error"] = f"{type(inner).__name__}: {inner}"
        return out

    out["type"] = type(obj).__name__
    if isinstance(obj, dict):
        keys = list(obj.keys())
        out["n_keys"] = len(keys)
        out["keys"] = [str(k) for k in keys[:40]]
        first = obj[keys[0]] if keys else None
        out["value_type"] = type(first).__name__
        if isinstance(first, dict):
            out["value_keys"] = [str(k) for k in first]
    elif isinstance(obj, (list, tuple)):
        out["n_items"] = len(obj)
        out["item_type"] = type(obj[0]).__name__ if obj else None
    return out


def inventory(root: Path, tree_depth: int = 3) -> dict[str, Any]:
    """Everything a later ticket needs to know about the staged tree.

    Deliberately data-driven rather than a list of expected paths: the point of
    the ingest is to find out what the bundle contains, so asserting a layout
    here would only hide the answer. Returns directory sizes to `tree_depth`,
    a probe of every pickle, and the header plus row count of every csv small
    enough to be a results or metadata table.
    """
    root = Path(root)
    print(f"  inventory: walking {root}", flush=True)
    tree: dict[str, dict[str, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        key = str(Path(*rel.parts[:tree_depth]).parent if len(rel.parts) > tree_depth else rel.parent)
        entry = tree.setdefault(key, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += path.stat().st_size

    pkl_paths = sorted(root.rglob("*.pkl"))
    print(f"  inventory: probing {len(pkl_paths)} pickles", flush=True)
    pickles = {str(p.relative_to(root)): _probe_pickle(p) for p in pkl_paths}

    csvs: dict[str, dict[str, Any]] = {}
    csv_paths = sorted(root.rglob("*.csv"))
    print(f"  inventory: reading {len(csv_paths)} csv headers", flush=True)
    for p in csv_paths:
        rel = str(p.relative_to(root))
        size = p.stat().st_size
        info: dict[str, Any] = {"bytes": size}
        # Per-assay DMS files run to hundreds of thousands; only summary and
        # metadata tables are small, and only those are worth reading here.
        if size < 8 << 20:
            with p.open(errors="replace") as f:
                header = f.readline().strip()
                info["header"] = header[:400]
                info["rows"] = sum(1 for _ in f)
        csvs[rel] = info

    return {
        "root": str(root),
        "total_files": sum(e["files"] for e in tree.values()),
        "total_bytes": sum(e["bytes"] for e in tree.values()),
        "tree": dict(sorted(tree.items())),
        "pickles": pickles,
        "csvs": csvs,
    }


def write_manifest(root: Path, entries: list[dict[str, Any]], git_sha: str | None) -> Path:
    """Record what populated this tree, inside the tree itself.

    A Volume is the one piece of Modal-side state this repo does not rebuild
    from a definition, so a later reader has no way to tell which ingest
    produced the bytes they are reading unless the tree says so. Doubles as the
    completion marker: no manifest means the ingest did not finish.
    """
    from datetime import datetime, timezone

    path = root / "_manifest.json"
    existing = json.loads(path.read_text()) if path.exists() else {"ingests": []}
    existing["ingests"].append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "sources": entries,
        }
    )
    path.write_text(json.dumps(existing, indent=2))
    return path
