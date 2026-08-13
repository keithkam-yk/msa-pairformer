"""The Lightning training package: does it run, and is the aggregation right?

Two claims are worth testing here, and they are not the same claim.

The first is that the code works. `training/` is new code -- phase 5 replaced
`training_utils.py` rather than moving it -- so nothing else in the suite
touches it, and "it imports" is not evidence. `test_trainer_*` therefore runs a
real `Trainer` over a real `MSAPairformer` for one training step and one
validation step, on CPU, at a 29k-parameter configuration rather than the
released 111M one.

The second is the reason the dependency exists at all. The deleted
`GradAccumLossTracker` kept `total_loss / total_tokens` and
`total_correct / total_tokens`, which is a token-weighted mean; it was replaced
because it never all-reduced those sums, not because a mean over batches would
have done. So the tests below construct two batches with *different* token
counts and assert the accumulated value is the token-weighted one. The
distinction is the whole point: with 9 of 10 correct in one batch and 0 of 2 in
the next, the mean of the per-batch means is 0.45 and the answer is 0.75. A
test built on two equal-sized batches passes under either implementation and
proves nothing.

The DDP half of the claim -- that `compute()` synchronises across ranks -- is
torchmetrics' own and is not re-tested here. What is tested is that this code
routes its numbers through `update`/`compute` rather than through Python
attributes, which is what makes that guarantee reachable.

Requires the training extra:

    uv sync --extra training
"""

import numpy as np
import pytest
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from bench.data import synthetic_batch
from bench.step import triangle_path

try:
    from lightning import Trainer

    from msa_pairformer.nn.model import MSAPairformer
    from msa_pairformer.training.datamodule import (
        LoaderSpec,
        MaskingSpec,
        MSADataModule,
        SplitSizes,
        SplitSpec,
        TrRosettaContactDataModule,
    )
    from msa_pairformer.training.module import MSAPairformerModule

    TRAINING_EXTRA_PRESENT = True
except ImportError:  # pragma: no cover - exercised only without the extra
    TRAINING_EXTRA_PRESENT = False

# Not a bare `return`: `pyproject.toml` escalates PytestReturnNotNoneWarning to
# an error precisely because a hand-rolled skip reads as green while checking
# nothing.
pytestmark = pytest.mark.skipif(
    not TRAINING_EXTRA_PRESENT,
    reason="needs the training extra (lightning, torchmetrics): uv sync --extra training",
)

# Small enough to train twice in a test process, and structurally a real model:
# two core layers, both triangle updates on, the same heads. The released
# configuration is 111M parameters and would make this file a benchmark.
TINY_MODEL_KWARGS = dict(
    dim_pairwise=16,
    dim_msa=16,
    core_module_kwargs=dict(
        depth=2,
        opm_kwargs=dict(
            dim_opm_hidden=4,
            outer_product_flavor="presoftmax_differential_attention",
            seq_attn=True,
            dim_qk=8,
            chunk_size=None,
            return_seq_weights=True,
            lambda_init=None,
            eps=1e-32,
        ),
        pwa_kwargs=dict(heads=2, dim_head=8, dropout=0.0, dropout_type="row"),
        pairwise_block_kwargs=dict(
            dropout_row_prob=0,
            dropout_col_prob=0,
            tri_mult_dim_hidden=None,
            use_triangle_updates=True,
            use_pair_updates=False,
        ),
    ),
    relative_position_encoding_kwargs=dict(r_max=8, s_max=2),
    contact_layer=0,
    confind_contact_layer=1,
    potts_layer_idx=0,
)

DEPTH, CROP = 4, 6

# Every name `training/module.py` logs. Pinned as a set rather than checked for
# membership: a renamed metric silently breaks every downstream dashboard and
# every resumed comparison, and an *added* one is a decision that should be
# made here rather than discovered in a log.
EXPECTED_METRICS = {
    "train/loss",
    "train/loss_step",
    "train/perplexity",
    "train/accuracy",
    "val/loss",
    "val/perplexity",
    "val/accuracy",
}

# Small vocabulary for the scripted-logits tests below. Nothing reads the
# tokens, so the real 26 would only make the expected values harder to write.
VOCAB = 4


class ScriptedModel(torch.nn.Module):
    """Reads its logits out of the batch, with `MSAPairformer`'s output shape.

    The token-weighting assertions need a known number of correct predictions
    across batches of different sizes, which a randomly initialised model
    cannot give. This substitutes for the network and for nothing else: the
    indexing, the loss and the metric updates under test are all
    `MSAPairformerModule`'s own.

    The logits travel in the first `VOCAB` channels of `msas_onehot`, which is
    what the module hands to the model as `msa`. That keeps the stub stateless,
    so a `DataLoader` can yield two differently scripted batches to one model --
    which is what the gradient-accumulation test needs.

    `bias` is zeros, so it changes no logit, and it exists so that the loss has
    a parameter to be differentiated against and `configure_optimizers` has
    something to optimise.
    """

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(VOCAB))

    def forward(self, msa, **_):
        return {"logits": msa[..., :VOCAB] + self.bias}


def scripted_batch(rows, depth, crop):
    """One collated batch whose masked positions are exactly `rows`.

    `rows` is a list of per-token logit vectors. Targets are all class 0, so a
    row is "correct" when its largest entry is the first one.
    """
    onehot = torch.zeros(1, depth, crop, 28)
    onehot[..., :VOCAB] = torch.tensor(rows, dtype=torch.float32).reshape(1, depth, crop, VOCAB)
    return {
        "msas": torch.zeros(1, depth, crop, dtype=torch.long),
        "msas_onehot": onehot,
        "mask": None,
        "msa_mask": None,
        "full_mask": None,
        "pairwise_mask": None,
        "masked_idx": torch.arange(depth * crop),
    }


# Nine correct out of ten, then zero out of two. Per-batch means: 0.9 and 0.0.
# Token-weighted: 9/12. The two are far enough apart that no tolerance choice
# can confuse them.
BATCH_A_ROWS = [[2.0, 0.0, 0.0, 0.0]] * 9 + [[0.0, 3.0, 0.0, 0.0]]
BATCH_B_ROWS = [[0.0, 1.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]]
BATCH_SHAPES = ((BATCH_A_ROWS, 2, 5), (BATCH_B_ROWS, 1, 2))
TOKEN_WEIGHTED_ACCURACY = 9 / 12
BATCH_MEAN_ACCURACY = (0.9 + 0.0) / 2

# The same twelve tokens as one sequence. Every expected value below is derived
# from this rather than written as a literal, so the arithmetic under test is
# `torch`'s and not a number someone typed.
EVERY_ROW = torch.tensor(BATCH_A_ROWS + BATCH_B_ROWS)
TOKEN_WEIGHTED_LOSS = float(cross_entropy(EVERY_ROW, torch.zeros(12, dtype=torch.long)))


@pytest.fixture(scope="module")
def tiny_model():
    """Built under the vanilla triangle path, like every other model in the suite.

    `conftest.py` forces it session-wide; entering it again here is harmless and
    keeps the construction honest if this file is ever run alone.
    """
    with triangle_path(False):
        torch.manual_seed(0)
        return MSAPairformer(**TINY_MODEL_KWARGS)


def cpu_trainer(**kwargs):
    """A `Trainer` that writes nothing anywhere.

    `logger=False` is the test's choice, not the module's: `training/`
    configures no logger at all (§10 decision 2), so the caller's `Trainer`
    decides where `self.log` goes -- including nowhere.
    """
    return Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        **kwargs,
    )


@pytest.fixture(scope="module")
def fast_dev_run(tiny_model):
    """One training step and one validation step, end to end, on CPU.

    Returns the trainer and the weights as they were before `fit`, so a test
    can ask whether the optimiser actually moved them.
    """
    module = MSAPairformerModule(model=tiny_model, checkpoint_triangles=False)
    before = {name: tensor.detach().clone() for name, tensor in module.model.state_dict().items()}
    trainer = cpu_trainer(fast_dev_run=True)
    trainer.fit(
        module,
        train_dataloaders=DataLoader([synthetic_batch(1, DEPTH, CROP, seed=0)], batch_size=None),
        val_dataloaders=DataLoader([synthetic_batch(1, DEPTH, CROP, seed=1)], batch_size=None),
    )
    return trainer, before


@pytest.fixture
def scripted_module():
    """A module wrapping `ScriptedModel`, with metrics sized to `VOCAB`.

    `num_classes` is explicit here because `ScriptedModel` has no `lm_head` to
    infer it from -- which is also the case the explicit argument exists for.
    """
    return MSAPairformerModule(model=ScriptedModel(), num_classes=VOCAB)


def test_trainer_runs_a_step_and_produces_a_finite_loss(fast_dev_run):
    """The step ran, and it produced a number an optimiser could use."""
    trainer, _ = fast_dev_run
    losses = [trainer.callback_metrics[name] for name in ("train/loss", "val/loss")]
    assert all(torch.isfinite(loss) for loss in losses), losses
    assert all(loss > 0 for loss in losses), losses


def test_the_training_step_actually_updates_the_weights(fast_dev_run):
    """A finite loss is not evidence of training.

    Every assertion above passes against a module whose optimiser never steps
    -- a forward pass alone produces all of them. This is the one that says the
    backward pass ran, `configure_optimizers` was honoured, and the gradient
    reached the model rather than stopping at a detached tensor.
    """
    trainer, before = fast_dev_run
    after = trainer.lightning_module.model.state_dict()
    changed = [name for name, tensor in before.items() if not torch.equal(tensor, after[name])]
    assert len(changed) > len(before) / 2, f"only {len(changed)} of {len(before)} tensors moved"


def test_metrics_are_logged_under_stable_names(fast_dev_run):
    trainer, _ = fast_dev_run
    assert set(trainer.callback_metrics) == EXPECTED_METRICS


def test_perplexity_is_the_exponential_of_the_loss(fast_dev_run):
    """A cross-check that the two metrics describe the same predictions.

    They are computed by different code -- `CrossEntropyLoss` through
    `MeanMetric`, and torchmetrics' own softmax and gather -- over the same
    tokens, so agreement is evidence that neither is being fed the wrong
    tensor. It is also what pins torchmetrics' input contract: `Perplexity`
    normalises internally, so handing it log-probabilities instead of logits
    would still produce a finite, plausible, wrong number.
    """
    trainer, _ = fast_dev_run
    for stage in ("train", "val"):
        loss = trainer.callback_metrics[f"{stage}/loss"]
        perplexity = trainer.callback_metrics[f"{stage}/perplexity"]
        assert perplexity == pytest.approx(float(torch.exp(loss)), rel=1e-5)


def test_accumulated_accuracy_is_token_weighted_not_the_mean_of_batch_means(scripted_module):
    """The assertion the old tracker's replacement has to survive."""
    for rows, depth, crop in BATCH_SHAPES:
        scripted_module._shared_step(scripted_batch(rows, depth, crop), "train")

    accumulated = float(scripted_module.train_accuracy.compute())
    assert accumulated == pytest.approx(TOKEN_WEIGHTED_ACCURACY)
    assert accumulated != pytest.approx(BATCH_MEAN_ACCURACY)


def test_accumulated_loss_is_token_weighted_not_the_mean_of_batch_means(scripted_module):
    """`MeanMetric` is updated with the token count as its weight.

    Without the weight it would average two batch means, which for a 10-token
    batch and a 2-token batch is not the mean over the 12 tokens.
    """
    batch_losses = []
    for rows, depth, crop in BATCH_SHAPES:
        loss, n_tokens = scripted_module._shared_step(scripted_batch(rows, depth, crop), "train")
        batch_losses.append(float(loss.detach()))
        assert n_tokens == depth * crop

    unweighted = sum(batch_losses) / len(batch_losses)
    assert float(scripted_module.train_loss.compute()) == pytest.approx(TOKEN_WEIGHTED_LOSS, rel=1e-6)
    assert TOKEN_WEIGHTED_LOSS != pytest.approx(unweighted, rel=1e-3)


def test_accumulated_perplexity_is_token_weighted(scripted_module):
    """Perplexity accumulates summed log-likelihood over a token count, so the
    same weighting has to fall out of it -- and it comes from torchmetrics'
    state rather than from this repository's arithmetic."""
    for rows, depth, crop in BATCH_SHAPES:
        scripted_module._shared_step(scripted_batch(rows, depth, crop), "train")

    expected = float(torch.exp(torch.tensor(TOKEN_WEIGHTED_LOSS)))
    assert float(scripted_module.train_perplexity.compute()) == pytest.approx(expected, rel=1e-5)


def test_lightning_aggregates_across_accumulate_grad_batches(scripted_module):
    """The claim §5.2 makes for deleting `GradAccumStatTracker`, through a Trainer.

    The three tests above drive `_shared_step` directly, so they check the
    metric objects and not Lightning's aggregation over an accumulation window.
    That aggregation is the whole argument for deleting a class, so it is
    pinned here: two micro-batches of different token counts inside one
    optimiser step, and the epoch value has to be the token-weighted one.

    `ScriptedModel.bias` is only updated when the window closes, after both
    forward passes, so the second micro-batch sees the same logits it would
    have seen alone.
    """
    batches = [scripted_batch(rows, depth, crop) for rows, depth, crop in BATCH_SHAPES]
    trainer = cpu_trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=0, accumulate_grad_batches=2)
    trainer.fit(scripted_module, train_dataloaders=DataLoader(batches, batch_size=None))

    assert trainer.global_step == 1, "the two micro-batches should be one optimiser step"
    assert float(trainer.callback_metrics["train/loss"]) == pytest.approx(TOKEN_WEIGHTED_LOSS, rel=1e-6)
    assert float(trainer.callback_metrics["train/accuracy"]) == pytest.approx(TOKEN_WEIGHTED_ACCURACY)
    assert float(trainer.callback_metrics["train/accuracy"]) != pytest.approx(BATCH_MEAN_ACCURACY)


def test_the_three_stages_do_not_share_metric_state(scripted_module):
    """One set per stage, so a validation batch cannot move a training number.

    Test has its own set rather than borrowing validation's, which is a
    naming decision as much as a state one: sharing would publish a test
    result under `val/accuracy`.
    """
    scripted_module._shared_step(scripted_batch(BATCH_B_ROWS, 1, 2), "val")

    assert float(scripted_module.val_accuracy.compute()) == pytest.approx(0.0)
    assert scripted_module.train_accuracy.update_count == 0
    assert scripted_module.test_accuracy.update_count == 0


def test_a_none_batch_is_skipped_rather_than_crashing(scripted_module):
    """`CollateAFBatch` returns None when every alignment is too shallow."""
    assert scripted_module.training_step(None, 0) is None
    assert scripted_module.validation_step(None, 0) is None


def paths(n=20):
    """Fake alignment paths. Nothing opens them: `MSADataset.__init__` stores
    the list and the a3m parsing happens in `__getitem__`, so the DataModule
    can be tested without a fixture directory or an `hhfilter` binary."""
    return [f"alignment_{i}.a3m" for i in range(n)]


def test_setup_splits_every_path_exactly_once():
    datamodule = MSADataModule(paths(), sizes=SplitSizes(train_prop=0.5, val_size=5))
    datamodule.setup("fit")

    sizes = {name: len(idx) for name, idx in datamodule.splits.items()}
    assert sizes == {"train": 10, "val": 5, "test": 5}
    assert sorted(np.concatenate(list(datamodule.splits.values())).tolist()) == list(range(20))


def test_the_split_is_reproducible_and_seed_dependent():
    """Reproducible from the seed, and from nothing else: the split draws from
    its own Generator, so it does not depend on -- or disturb -- global state."""
    torch.manual_seed(999)
    first = MSADataModule(paths(), sizes=SplitSizes(seed=1))
    first.setup("fit")
    torch.manual_seed(0)
    again = MSADataModule(paths(), sizes=SplitSizes(seed=1))
    again.setup("fit")
    other = MSADataModule(paths(), sizes=SplitSizes(seed=2))
    other.setup("fit")

    assert first.splits["train"].tolist() == again.splits["train"].tolist()
    assert first.splits["train"].tolist() != other.splits["train"].tolist()


def test_each_split_gets_its_own_spec():
    """The duplication the 33 parameters encoded, now expressed once per split."""
    datamodule = MSADataModule(
        paths(),
        train=SplitSpec(max_seq_length=312, max_msa_depth=256, batch_size=2, shuffle=True),
        val_test=SplitSpec(max_seq_length=64, max_msa_depth=32, batch_size=1),
        masking=MaskingSpec(mask_prob=0.25),
    )
    datamodule.setup("fit")

    train, val, test = (datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader())
    assert (train.batch_size, val.batch_size, test.batch_size) == (2, 1, 1)
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    # The collate is built from the same spec as the dataset it batches, which
    # is what stops a val loader from padding to the train crop.
    assert (train.collate_fn.max_seq_length, val.collate_fn.max_seq_length) == (312, 64)
    assert train.collate_fn.mask_prob == val.collate_fn.mask_prob == 0.25
    assert datamodule.datasets["test"].max_msa_depth == 32


def test_a_query_only_disagreement_is_refused(scripted_module):
    """The failure mode with no symptom, made into a failure.

    A collate corrupting query rows only, against a module reading every row,
    produces in-range indices against the wrong targets: no exception, a
    plausible loss, and a model trained on nothing. The check runs in
    `setup`, which is before the first batch is ever requested -- so this
    raises without touching the fake paths on disk.
    """
    datamodule = MSADataModule(paths(), masking=MaskingSpec(query_only=True))
    assert scripted_module.query_only is False

    with pytest.raises(ValueError, match="query_only disagrees"):
        cpu_trainer(fast_dev_run=True).fit(scripted_module, datamodule=datamodule)


def test_a_dataloader_before_setup_says_so():
    with pytest.raises(RuntimeError, match="call setup"):
        MSADataModule(paths()).train_dataloader()


def test_worker_only_loader_kwargs_appear_only_with_workers():
    """`persistent_workers=True` and `prefetch_factor` are errors at
    `num_workers=0`, which is why the old code could not run in one process."""
    assert LoaderSpec().kwargs() == {"num_workers": 0, "pin_memory": True}
    with_workers = LoaderSpec(num_workers=2).kwargs()
    assert with_workers == {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_trrosetta_datamodule_builds_both_loaders():
    """The second DataModule: same shape, no masking, splits given not derived."""
    datamodule = TrRosettaContactDataModule(
        [("a.a3m", "a.npz"), ("b.a3m", "b.npz")],
        [("c.a3m", "c.npz")],
        loader=LoaderSpec(num_workers=0, pin_memory=False),
    )
    datamodule.setup("fit")

    assert len(datamodule.datasets["train"]) == 2
    assert len(datamodule.datasets["val"]) == 1
    # Shuffling on train and not on validation is what the old
    # `init_trRosetta_contact_dataloaders` did, kept as the default.
    assert isinstance(datamodule.train_dataloader().sampler, RandomSampler)
    assert isinstance(datamodule.val_dataloader().sampler, SequentialSampler)
