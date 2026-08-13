"""Dataloader construction as configuration, not as a 33-parameter signature.

`init_dataloaders` took 33 parameters. Eight were one knob written twice --
`max_seq_length` and `max_seq_length_val_test`, `max_msa_depth` and
`max_msa_depth_val_test`, and so on for six more pairs -- so a caller had to
keep two parallel argument lists in the right order at every call site. The
duplication is not the interesting part of the configuration; the *split* is.
So one `SplitSpec` describes one split, and a DataModule holds two of them.

Three groups, and the grouping is the design:

* `SplitSpec` -- everything that can differ between train and validation: the
  crop, the depth, the token budget, the batch size, query selection, and
  whether the loader shuffles.
* `MaskingSpec` -- the MLM corruption, shared by every split. `init_dataloaders`
  shared it too, by passing the same five values to both collates.
* `LoaderSpec` -- `torch.utils.data.DataLoader` plumbing, shared by every split.

Four parameters of `init_dataloaders` are gone rather than regrouped:
`n_append_random`, `n_append_random_val_test`, `shifted_random`,
`scramble_seq_perc_*` and `scramble_col_perc_*`. They name a `MSADataset` and a
`CollateAFBatch` that no longer exist -- neither class accepts any of them
today, so `init_dataloaders` raised `TypeError` before it built its first
dataset:

    MSADataset.__init__() got an unexpected keyword argument 'n_append_random'

Carrying them forward would mean carrying forward the same `TypeError`. They
are listed here so the loss is recorded rather than silent: if the appending
and scrambling augmentations come back, they come back in `MSADataset` first
and in `SplitSpec` second.

Seeding is deliberately local. `init_dataloaders` called `torch.manual_seed`,
`np.random.seed` and `random.seed` as a side effect of building loaders, which
reseeded the model initialisation of whatever ran next. The split here draws
from its own `numpy.random.Generator`, so it is reproducible without touching
global state; process-wide seeding is the caller's, through
`lightning.seed_everything(seed, workers=True)` -- which, unlike the deleted
`set_seed`, also seeds the dataloader workers this module can spawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from msa_pairformer import aa2tok_d
from msa_pairformer.data.msa_datasets import (
    CollateAFBatch,
    CollatetrRosettaContactMSABatch,
    MSADataset,
    trRosettaContactMSADataset,
)


@dataclass(frozen=True)
class SplitSpec:
    """How one split reads its alignments, and how it batches them.

    Defaults are `init_dataloaders`' own where it had them, and `MSADataset`'s
    where it did not. `shuffle` defaults to False because that is what the old
    train loader did -- it passed `shuffle=False` for train, val and test
    alike. That is preserved rather than corrected, and exposed rather than
    buried: a pre-training run almost certainly wants `shuffle=True` on train,
    but silently flipping it here would change what a resumed run trains on.
    """

    max_seq_length: int = 1024
    max_msa_depth: int = 1024
    min_msa_depth: int = 4
    max_tokens: int = 2**17
    batch_size: int = 1
    random_query: bool = False
    min_query_coverage: float = 0.8
    shuffle: bool = False


@dataclass(frozen=True)
class MaskingSpec:
    """The MLM corruption `CollateAFBatch` applies, shared by every split.

    `mask_prob` is `init_dataloaders`' `noising_rate` under the name the
    collate itself uses; the rename removes a translation step that existed
    only in the caller's head. `query_only` belongs here rather than in
    `SplitSpec` because it decides which positions are corrupted *and* which
    logits the loss reads, so it has to agree with `MSAPairformerModule`'s own
    `query_only` -- which `MSAPairformerModule.setup` now checks, because a
    disagreement produces in-range indices against the wrong targets and no
    error at all.
    """

    mask_prob: float = 0.15
    mask_ratio: float = 0.8
    mutate_ratio: float = 0.1
    keep_ratio: float = 0.1
    mutate_pssm: bool = False
    query_only: bool = False


@dataclass(frozen=True)
class LoaderSpec:
    """`DataLoader` plumbing, shared by every split.

    `num_workers=0` is the default because it is the only setting that works
    everywhere, including inside a test process. It also forces the shape of
    `kwargs()`: `persistent_workers=True` and `prefetch_factor` are errors when
    there are no workers, so they are emitted only when there are. The old code
    passed all three unconditionally and therefore could not run single-process
    at all.
    """

    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    def kwargs(self) -> dict[str, Any]:
        loader_kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return loader_kwargs


@dataclass(frozen=True)
class SplitSizes:
    """The train/validation/test proportions, as one value.

    `val_size` is a count and `train_prop` a fraction, which is how
    `init_dataloaders` had it: validation is a fixed budget and test is
    whatever remains.
    """

    train_prop: float = 0.8
    val_size: int = 100
    seed: int = 42


class MSADataModule(LightningDataModule):
    """MLM pre-training data: split once in `setup`, three loaders out.

    The whole of `init_dataloaders` is here, and so is the part it could not
    express: `setup` runs once per process under a `Trainer` and is the
    documented place for the split, so the indices cannot drift between the
    train loader and the validation loader the way they can when three
    functions each re-derive them.
    """

    def __init__(
        self,
        msa_paths,
        *,
        train: SplitSpec | None = None,
        val_test: SplitSpec | None = None,
        masking: MaskingSpec | None = None,
        loader: LoaderSpec | None = None,
        sizes: SplitSizes | None = None,
    ):
        super().__init__()
        self.msa_paths = np.asarray(msa_paths, dtype=object)
        self.train_spec = train or SplitSpec()
        self.val_test_spec = val_test or SplitSpec()
        self.masking = masking or MaskingSpec()
        self.loader = loader or LoaderSpec()
        self.sizes = sizes or SplitSizes()
        self.splits: dict[str, np.ndarray] = {}
        self.datasets: dict[str, Dataset] = {}

    def split_indices(self) -> dict[str, np.ndarray]:
        """One permutation, sliced three ways.

        `init_dataloaders` drew three `np.random.choice` samples and removed
        each from the pool with `setdiff1d`, which is a permutation written
        the long way -- and which silently reorders the remainder, so the test
        split was sorted while the other two were not. A single permutation
        gives the same disjointness with the same distribution and one
        obvious invariant: the three slices concatenate back to the whole.
        """
        n_total = len(self.msa_paths)
        n_train = int(self.sizes.train_prop * n_total)
        n_val = min(self.sizes.val_size, n_total - n_train)
        order = np.random.default_rng(self.sizes.seed).permutation(n_total)
        return {
            "train": order[:n_train],
            "val": order[n_train : n_train + n_val],
            "test": order[n_train + n_val :],
        }

    def _dataset(self, indices: np.ndarray, spec: SplitSpec) -> MSADataset:
        return MSADataset(
            msa_dir=None,
            msa_paths=self.msa_paths[indices],
            max_seq_length=spec.max_seq_length,
            max_msa_depth=spec.max_msa_depth,
            max_tokens=spec.max_tokens,
            min_depth=spec.min_msa_depth,
            random_query=spec.random_query,
            min_query_coverage=spec.min_query_coverage,
        )

    def _collate(self, spec: SplitSpec) -> CollateAFBatch:
        return CollateAFBatch(
            max_seq_length=spec.max_seq_length,
            max_seq_depth=spec.max_msa_depth,
            min_seq_depth=spec.min_msa_depth,
            mask_prob=self.masking.mask_prob,
            mask_ratio=self.masking.mask_ratio,
            mutate_ratio=self.masking.mutate_ratio,
            keep_ratio=self.masking.keep_ratio,
            mutate_pssm=self.masking.mutate_pssm,
            query_only=self.masking.query_only,
        )

    def setup(self, stage: str | None = None) -> None:
        """Build all three splits whatever the stage.

        Constructing an `MSADataset` opens no file -- the a3m parsing and the
        `hhfilter` subprocess happen in `__getitem__` -- so there is nothing to
        gain by building them lazily per stage, and something to lose: a
        `test_dataloader()` call after a fit-only `setup("fit")` would fail on
        a missing attribute rather than return the split the seed defines.
        """
        self.splits = self.split_indices()
        self.datasets = {
            "train": self._dataset(self.splits["train"], self.train_spec),
            "val": self._dataset(self.splits["val"], self.val_test_spec),
            "test": self._dataset(self.splits["test"], self.val_test_spec),
        }

    def _dataloader(self, name: str, spec: SplitSpec) -> DataLoader:
        if name not in self.datasets:
            raise RuntimeError(f"no {name} dataset: call setup() first (a Trainer does this for you)")
        return DataLoader(
            self.datasets[name],
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            collate_fn=self._collate(spec),
            **self.loader.kwargs(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train", self.train_spec)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val", self.val_test_spec)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test", self.val_test_spec)


class TrRosettaContactDataModule(LightningDataModule):
    """Contact supervision from trRosetta, as the second DataModule.

    Same shape as `MSADataModule`, different dataset and no masking: the
    targets are contact maps read from the `.npz` beside each alignment, so
    there is nothing to corrupt. The splits arrive pre-made -- `init_trRosetta_
    contact_dataloaders` took two path lists and never split anything -- so
    this one has no `SplitSizes` and no `test_dataloader`.

    Its predecessor had a second break past the one in the module docstring,
    and an independent one: it asked for `aa2tok_d['PAD']`, a key renamed to
    `'<pad>'`, so it raised `KeyError` even with the dataset arguments fixed.
    """

    def __init__(
        self,
        train_paired_paths,
        val_paired_paths,
        *,
        train: SplitSpec | None = None,
        val: SplitSpec | None = None,
        loader: LoaderSpec | None = None,
    ):
        super().__init__()
        self.train_paired_paths = list(train_paired_paths)
        self.val_paired_paths = list(val_paired_paths)
        # Three defaults, each carried from the old function rather than from
        # `SplitSpec`. It shuffled the training loader here and not in
        # `init_dataloaders`, so that survives as a default instead of a literal
        # buried in a `DataLoader` call. `min_msa_depth` was 8, not 4. And
        # `min_query_coverage` is 0.9 because the old function never passed the
        # argument at all, which left `trRosettaContactMSADataset`'s own 0.9
        # default in force -- `SplitSpec` defaults to `MSADataset`'s 0.8, so
        # taking the dataclass default here would have moved the threshold
        # silently. A caller who builds their own `SplitSpec` gets 0.8, which
        # is then their decision rather than this module's.
        self.train_spec = train or SplitSpec(min_msa_depth=8, min_query_coverage=0.9, shuffle=True)
        self.val_spec = val or SplitSpec(min_msa_depth=8, min_query_coverage=0.9)
        self.loader = loader or LoaderSpec()
        self.datasets: dict[str, Dataset] = {}

    def _dataset(self, paired_paths: list, spec: SplitSpec) -> trRosettaContactMSADataset:
        return trRosettaContactMSADataset(
            paired_paths_l=paired_paths,
            max_seq_length=spec.max_seq_length,
            max_msa_depth=spec.max_msa_depth,
            min_msa_depth=spec.min_msa_depth,
            max_tokens=spec.max_tokens,
            random_query=spec.random_query,
            min_query_coverage=spec.min_query_coverage,
        )

    def _collate(self, spec: SplitSpec) -> CollatetrRosettaContactMSABatch:
        return CollatetrRosettaContactMSABatch(
            max_seq_length=spec.max_seq_length,
            max_seq_depth=spec.max_msa_depth,
            min_seq_depth=spec.min_msa_depth,
            pad_tok=aa2tok_d["<pad>"],
        )

    def setup(self, stage: str | None = None) -> None:
        self.datasets = {
            "train": self._dataset(self.train_paired_paths, self.train_spec),
            "val": self._dataset(self.val_paired_paths, self.val_spec),
        }

    def _dataloader(self, name: str, spec: SplitSpec) -> DataLoader:
        if name not in self.datasets:
            raise RuntimeError(f"no {name} dataset: call setup() first (a Trainer does this for you)")
        return DataLoader(
            self.datasets[name],
            batch_size=spec.batch_size,
            shuffle=spec.shuffle,
            collate_fn=self._collate(spec),
            **self.loader.kwargs(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train", self.train_spec)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val", self.val_spec)
