"""The MLM training step, with the aggregation given to torchmetrics.

`training_utils.py` carried three tracker classes. All three are gone here, and
only one of them is a question of idiom:

* `GradAccumStatTracker` summed values and divided by a step count. Lightning
  already averages a logged metric across `accumulate_grad_batches`, so the
  class existed only because the loop was hand-written.
* `LossTracker` kept an exponential moving average per category. Smoothing a
  curve is the reader's job at the far end of the log, not the model's, so the
  step loss is logged raw with `on_step=True` and whatever logger the caller's
  `Trainer` carries can smooth it.
* `GradAccumLossTracker` is the one that was not merely redundant. It kept
  `total_loss / total_tokens` and `total_correct / total_tokens` in plain
  Python attributes and never all-reduced them. On one GPU that is correct; on
  more than one it reports rank zero's metrics as the run's, quietly. Every
  metric below synchronises across ranks inside `compute()`, which is the
  actual reason for the dependency -- not that torchmetrics is idiomatic.

The token weighting survives the swap, and it is not incidental. Batches carry
different numbers of masked positions, so the mean of the per-batch means is not
the accuracy over the tokens. `MulticlassAccuracy(average="micro")` accumulates
hits and a count; `Perplexity` accumulates summed negative log-likelihood and a
count; the loss goes through `MeanMetric` weighted by the token count. All three
therefore give the same number the old `total / total_tokens` did -- across
accumulation steps, across the epoch, and now across ranks.

`evaluate_prediction` took `device` and `criterion` as arguments. A
`LightningModule` owns both, so `_shared_step` takes neither.

No logger is configured here, deliberately. `self.log` writes to whatever the
caller's `Trainer` was given, including nothing at all.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity

from msa_pairformer import MSAPairformer

# The stages that log, and the prefix each one logs under. Test gets its own
# set rather than borrowing validation's: the state could be shared, since
# `Trainer.test` and `Trainer.validate` never run at the same time, but the
# *names* would be shared with it, and a `val/accuracy` that is sometimes a
# test result is a log a reader cannot interpret.
TRAIN, VAL, TEST = "train", "val", "test"


def _logit_dim(model: Module) -> int:
    """How many classes the language-model head emits.

    Needed at construction, because a torchmetrics classification metric sizes
    its state up front. Read off the head rather than defaulted to 26, so a
    model built with a different `dim_logits` does not get a metric that
    silently ignores its last classes.
    """
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None:
        raise ValueError(
            "cannot infer the number of token classes from this model: it has no `lm_head.weight`. "
            "Pass `num_classes=` explicitly."
        )
    return int(weight.shape[0])


class MSAPairformerModule(LightningModule):
    """Masked-language-model training for `MSAPairformer`.

    The model is wrapped, not subclassed, and that is the constraint from
    section 5.3 of the layout document rather than a preference: `bench/`
    builds the same `MSAPairformer` and drives it directly, with its own
    compilation and its own gradient accumulation. Nothing here reaches back
    into `nn/`, so the benchmark harness never has to know a `Trainer` exists.

    Pass `model` to wrap an instance -- which is what a test with a small
    configuration wants -- or `model_kwargs` to have one built.

    **Only the `model_kwargs` path round-trips through a checkpoint.**
    `save_hyperparameters` ignores `model`, so a checkpoint written from the
    `model=` path records `model_kwargs=None`, and `load_from_checkpoint` then
    rebuilds the *default* network and raises on the missing and unexpected
    `state_dict` keys. It fails loudly rather than restoring the wrong weights,
    but it fails: a run that intends to be resumable has to describe its model
    with `model_kwargs`, and load the weights into that.
    """

    def __init__(
        self,
        model: Module | None = None,
        *,
        model_kwargs: dict[str, Any] | None = None,
        num_classes: int | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.98),
        query_only: bool = False,
        checkpoint_triangles: bool = True,
    ):
        super().__init__()
        if model is not None and model_kwargs is not None:
            raise ValueError("pass `model` or `model_kwargs`, not both: the kwargs would be ignored")
        self.save_hyperparameters(ignore=["model"])
        self.model = model if model is not None else MSAPairformer(**(model_kwargs or {}))
        self.query_only = query_only
        self.checkpoint_triangles = checkpoint_triangles
        # Mean over the batch's masked positions, which is what backpropagates.
        # The token weighting that makes the *reported* number right is applied
        # in the metrics below, not here: scaling the gradient by a batch's
        # token count would change the optimisation, not just the logging.
        self.criterion = CrossEntropyLoss()

        n_classes = num_classes if num_classes is not None else _logit_dim(self.model)
        # One set per stage, written out rather than built in a loop, because
        # each of these is a submodule: assigning them by name is what puts them
        # in `state_dict` and what moves their state to the accelerator with the
        # rest of the module. Sharing one set between two stages would mix them
        # into a single number, since `compute()` reads whatever `update()`
        # accumulated since the last reset.
        self.train_loss, self.val_loss, self.test_loss = MeanMetric(), MeanMetric(), MeanMetric()
        self.train_perplexity, self.val_perplexity, self.test_perplexity = Perplexity(), Perplexity(), Perplexity()
        self.train_accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.val_accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.test_accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")

    def setup(self, stage: str) -> None:
        """Refuse a DataModule whose masking disagrees with this module.

        `query_only` is set in two places -- `MaskingSpec.query_only` decides
        which positions the collate corrupts and how it writes `masked_idx`,
        and this module's `query_only` decides which logits and which targets
        that index is read against. Agreement was a request in prose until now,
        and disagreement has no symptom: the indices stay in range, the loss
        stays finite, and the model trains against the wrong targets.

        Checked here because this is the first hook where both objects exist.
        Only a `Trainer` with a DataModule reaches the comparison; a run that
        passes dataloaders directly has no `MaskingSpec` to compare against.
        """
        masking = getattr(getattr(self._trainer, "datamodule", None), "masking", None)
        if masking is not None and masking.query_only != self.query_only:
            raise ValueError(
                f"query_only disagrees: the DataModule's MaskingSpec says {masking.query_only} and this "
                f"module says {self.query_only}. The collate and the loss would read different positions."
            )

    def _metrics(self, stage: str) -> tuple[MeanMetric, Perplexity, MulticlassAccuracy]:
        """The three metrics for one stage, in a fixed order."""
        return (
            getattr(self, f"{stage}_loss"),
            getattr(self, f"{stage}_perplexity"),
            getattr(self, f"{stage}_accuracy"),
        )

    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Logits for one collated batch.

        Five of `MSAPairformer.forward`'s defaults are wrong for training, for
        the reasons `bench/step.py:micro_step` sets out at length: the two
        contact heads run work MLM never reads, and the two `store_*_cpu` flags
        send the final representations to the host and back on the autograd
        path. `use_checkpointing_triangles` is the fifth, and it is on by
        default here because at the paper's shapes the alternative is an OOM --
        but only while a gradient is actually being built, since under
        `torch.no_grad` checkpointing buys nothing and warns.
        """
        results = self.model(
            msa=batch["msas_onehot"].float(),
            mask=batch["mask"],
            msa_mask=batch["msa_mask"],
            full_mask=batch["full_mask"],
            pairwise_mask=batch["pairwise_mask"],
            query_only=self.query_only,
            return_cb_contacts=False,
            return_confind_contacts=False,
            store_msa_repr_cpu=False,
            store_pairwise_repr_cpu=False,
            use_checkpointing_triangles=self.checkpoint_triangles and torch.is_grad_enabled(),
        )
        return results["logits"]

    def _masked_predictions(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """The logits and targets at the corrupted positions, both flattened.

        This is the half of `evaluate_prediction` that was about indexing.
        `masked_idx` indexes the flattened alignment, and when `query_only` is
        set the collate has already rewritten it to index the query row alone --
        so the target has to come from the same view the collate assumed.
        """
        masked_idx = batch["masked_idx"]
        if masked_idx is None:
            raise ValueError(
                "batch has no `masked_idx`: the collate ran with mask_prob=0, so there is nothing to predict"
            )
        logits = self(batch)
        pred = logits.reshape(-1, logits.shape[-1])[masked_idx]
        msas = batch["msas"]
        flat = msas[:, 0, :].flatten() if self.query_only else msas.reshape(-1)
        return pred.float(), flat[masked_idx].long()

    def _update_metrics(self, stage: str, pred: Tensor, target: Tensor, loss: Tensor) -> None:
        """Accumulate one batch into the stage's metrics, token-weighted.

        Kept separate from the `self.log` calls so the accumulation can be
        exercised -- and its weighting asserted -- without a `Trainer`.

        `Perplexity` wants `[batch, sequence, vocab]` and applies its own
        softmax, so the flat `[tokens, vocab]` selection goes in as a single
        sequence. It accumulates the summed log-likelihood and the token count
        separately, which is exactly why that reshaping is safe: the sequence
        axis carries no meaning to it beyond the count.
        """
        loss_metric, perplexity, accuracy = self._metrics(stage)
        loss_metric.update(loss.detach(), weight=target.numel())
        perplexity.update(pred.detach().unsqueeze(0), target.unsqueeze(0))
        accuracy.update(pred.detach(), target)

    def _log_metrics(self, stage: str) -> None:
        """Log the metric objects, not their current values.

        Lightning calls `compute()` at the epoch boundary and `reset()` after,
        and `compute()` is where the all-reduce happens. Logging
        `metric.compute()` here instead would publish a partial, rank-local
        number every step -- which is what the deleted trackers did.
        """
        loss_metric, perplexity, accuracy = self._metrics(stage)
        self.log(f"{stage}/loss", loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/perplexity", perplexity, on_step=False, on_epoch=True)
        self.log(f"{stage}/accuracy", accuracy, on_step=False, on_epoch=True)

    def _shared_step(self, batch: dict[str, Any], stage: str) -> tuple[Tensor, int]:
        """`evaluate_prediction`, less four of its arguments.

        Two go because a `LightningModule` owns them: `device` and `criterion`.

        The other two go because nothing produces them, and they are named here
        for the same reason `datamodule.py` names the parameters it dropped --
        so the narrowing is recorded rather than silent. `loss_weights`
        multiplied the per-token loss by a caller-supplied tensor, and
        `mean_reduction` then reduced it; both require a criterion built with
        `reduction="none"`, and no caller in this repository ever passed
        either. Per-token loss weighting is a real feature -- curriculum
        weighting, sequence weighting -- and when something wants it, it comes
        back as a field on this module and a `reduction="none"` criterion, not
        as two arguments threaded through every call site.
        """
        pred, target = self._masked_predictions(batch)
        loss = self.criterion(pred, target)
        self._update_metrics(stage, pred, target, loss)
        return loss, target.numel()

    def training_step(self, batch: dict[str, Any] | None, batch_idx: int) -> Tensor | None:
        """`None` batches are skipped, because the collate emits them.

        `CollateAFBatch` returns `None` when every alignment in the batch is
        shallower than `min_seq_depth`. Returning `None` from here is
        Lightning's documented way to skip a batch, so a shallow draw costs a
        step rather than a crash.
        """
        if batch is None:
            return None
        loss, n_tokens = self._shared_step(batch, TRAIN)
        # The raw per-step value, unsmoothed: this is the `LossTracker` EMA's
        # replacement, and the smoothing now happens where the curve is drawn.
        # `batch_size` is stated because the batch is a dict and Lightning
        # cannot infer it -- and the count that matters here is tokens, not
        # alignments.
        self.log(f"{TRAIN}/loss_step", loss, on_step=True, on_epoch=False, batch_size=n_tokens)
        self._log_metrics(TRAIN)
        return loss

    def validation_step(self, batch: dict[str, Any] | None, batch_idx: int) -> Tensor | None:
        if batch is None:
            return None
        loss, _ = self._shared_step(batch, VAL)
        self._log_metrics(VAL)
        return loss

    def test_step(self, batch: dict[str, Any] | None, batch_idx: int) -> Tensor | None:
        """Identical to validation, under its own three metric names."""
        if batch is None:
            return None
        loss, _ = self._shared_step(batch, TEST)
        self._log_metrics(TEST)
        return loss

    def configure_optimizers(self):
        """AdamW, and no schedule.

        `training_utils.py` had neither, so there is no schedule to port and
        none is invented here: a warmup and a decay are training-run policy,
        and they belong to the run that first needs them.
        """
        # Subscripted rather than dotted: `self.hparams` is a mapping that also
        # answers attribute access, and only the mapping half is declared, so
        # `hparams["lr"]` is the form a type checker can follow.
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            betas=self.hparams["betas"],
            weight_decay=self.hparams["weight_decay"],
        )
