# Module layout: analysis and proposal

This document examines the layout of the `msa_pairformer` package. It gives the
evidence for the structural problems, an organising principle that the
repository already contains, a target layout, and the constraints that any
change must respect.

The text uses ASD-STE100 Simplified Technical English. Sentences are short. Each
sentence gives one idea. The tenses are simple.

**This document proposes. It moves no files.** Section 9 gives the order of
work if you decide to do it.

Date: 12 August 2026. Repository commit: `ddcbc26`.

## Summary

The package has 20 modules, 7,694 lines, and an empty `__init__.py`. Because
the facade is empty, all 20 modules are public surface. Consumers import deep
paths, so no module can move without a break. This is the root cause. Every
other problem below grows from it.

| Problem | Evidence | Fix |
| --- | --- | --- |
| No facade; 20 public modules | `__init__.py` is 0 lines | Curate `__init__.py`. Additive. §2.1 |
| The tokenizer lives in the dataloader | 12 of 13 external imports of `dataset.py` want tokens or masks. One wants a `Dataset`. | Split `dataset.py`. §2.2 |
| `utils.py` is four unrelated modules | 695 lines, 21 functions, 4 concerns, 12 functions with no caller | Split by concern. §2.3 |
| Duplicated files | 2 duplicate pairs, one of them across the package boundary | Delete the copies. §2.4 |
| The training code is a hand-written loop | `init_dataloaders` takes 44 parameters. Three tracker classes re-implement metric aggregation, and one of them is not DDP-safe. | Rewrite on Lightning. §5 |
| `_run` and `write_log` discard information | The wrapper drops `returncode` and rewrites `CompletedProcess.stdout` into a list | Use the standard library. §6 |

The count of ten `utils` is correct. Five are in the package
(`utils.py`, `training_utils.py`, `proteingym_utils.py`,
`potts_diffpalm_utils.py`, `pairing_optimization/gumbel_sinkhorn_utils.py`).
Four are in `Cell_2026_analysis_notebooks/Figure5/utils/`. The tenth is that
directory itself.

**The proposal removes all ten.** Five go from the package. Five go from the
figure directory, which section 7 covers.

## 1. What is there now

The package, by size. These are the counts at `ddcbc26`, before any phase ran;
the sections below describe that starting point and are not updated as phases
land. Read the git history for the current state.

```
757  model.py                    734  dataset.py                  695  utils.py
485  pairwise_operations.py      386  outer_product.py            352  training_utils.py
321  chunk_layer.py              241  plotting.py                 174  core.py
123  regression.py                85  custom_typing.py             85  positional_encoding.py
 75  categorical_jacobian.py      38  data_downloader.py           32  potts_diffpalm_utils.py
620  proteingym_utils.py
pairing_optimization/: 880 mp_pdp.py, 711 msa_transformer_diffpalm.py, 465 mp_em.py,
                       184 datasets.py, 101 gumbel_sinkhorn_utils.py, 37 msa_parsing.py
```

There is one subpackage, `pairing_optimization/`. Everything else is flat.

## 2. The problems in the current package

### 2.1 The facade is empty, so nothing can move

`msa_pairformer/__init__.py` contains nothing. A user who wants the model must
write `from msa_pairformer.model import MSAPairformer`. A user who wants to
tokenise must write `from msa_pairformer.dataset import aa2tok_d`.

The result is that internal file names are the public API. The Cell 2026 figure
scripts import from nine different modules. `bench/` imports from five.
`tests/` imports from five. Renaming `regression.py` — which contains three
prediction heads and no regression — breaks all of them.

An empty `__init__.py` is not a neutral choice. It is a decision to publish
every file.

This matters more now than it did last week. The training package in section 5
will be the facade's first real consumer. Written against a facade it stays
insulated from later moves. Written against deep paths it adds a new set of
call sites that pin the current layout in place.

### 2.2 The tokenizer is inside the dataloader

`dataset.py` is 734 lines. It holds five different things:

1. The token vocabulary: `aa2tok_d`, `tok2aa_d`, `nTokenTypes`, `code2aa_d`,
   and the ESM cross-walk `esmtok_to_pairformertok_d`.
2. The `MSA` class, which reads a file from disk.
3. Masked-language-model masking: `msa_mlm`.
4. Tensor preparation: `prepare_msa_masks`, `prepare_inputs`, `onehot_msa`,
   `get_relative_positions`, `prep_molecule_feats`.
5. PyTorch `Dataset` and collate plumbing: `MSADataset`, `CollateAFBatch`,
   `trRosettaContactMSADataset`, `CollatetrRosettaContactMSABatch`.

Now look at who imports it. There are 13 external import sites in `.py` files,
and further sites in the figure notebooks:

| What they import | Sites |
| --- | --- |
| `aa2tok_d`, `prepare_msa_masks`, `tok2aa_d`, `esmtok_to_pairformertok_d`, `nTokenTypes`, `msa_mlm`, `MSA` | 12 |
| `MSADataset`, `CollateAFBatch`, `trRosettaContactMSADataset` | 1 |

The one site that imports the `Dataset` classes is `training_utils.py`. No
notebook imports them either. So today the dataloader half of `dataset.py` has
exactly one consumer, and that consumer is the code section 5 rewrites.

The cost is real, not stylistic. `dataset.py` imports `Bio.SeqIO`,
`scipy.spatial.distance.cdist`, `subprocess`, `tempfile` and `glob`, and it
shells out to `hhfilter` at line 330. A test that wants a token dictionary pays
for all of that. The tokenizer is a property of the model. It should sit next
to the model, not behind a module that runs subprocesses.

### 2.3 `utils.py` is four modules in one file

`utils.py` has 695 lines and 21 top-level functions, one of which is private.
They fall into four groups that share no data and no imports:

| Group | Functions | Dependency |
| --- | --- | --- |
| Structure file I/O | `get_coords`, `get_coords_cif`, `get_distance_matrix`, `write_chain_from_pdb`, `write_chain_from_cif`, `convert_cif_to_pdb` | `Bio.PDB` |
| Contact metrics | `get_p_at_k`, `get_p_at_l`, `compute_precision`, `compute_precisions`, `evaluate_contact_prediction`, `get_top_L_pairs`, `get_contacts` | `torch`, `numpy` |
| CONFIND subprocess wrappers | `run_confind`, `run_confind_mp`, `run_batch_confind`, `extract_confind_contacts`, `extract_homooligomeric_confind_contacts` | `subprocess` |
| Sequence weighting | `fit_seq_weight_mixture_model` | `sklearn` |

Two functions fit no group: `write_log`, and the private `_run` that wraps
`subprocess`. Section 6 covers them.

Twelve of the 20 public functions have no caller anywhere in the repository,
notebooks included. The count uses whole-word matching, so `compute_precision`
is not counted as a hit on `compute_precisions`.

The CONFIND cluster is one of the unused groups, and it stays. The concept is
live — `model.py` has `predict_confind_contacts` — so those five functions are
how a user produces its input. They are public API that happens to have no
in-repository caller. They go to `evaluate/confind.py` with a docstring that
says so.

`fit_seq_weight_mixture_model` is the orphan. It is sequence weighting, and
sequence weighting has three separate homes: this function,
`proteingym_utils.calc_weights_fast`, and
`Cell_2026_analysis_notebooks/Figure5/utils/weights.py`.

Renaming `utils.py` to `helpers.py` would fix nothing. The name is honest. The
file really is a bag. The bag is the problem.

### 2.4 Two duplicate pairs

**Inside the package.** `potts_diffpalm_utils.py` and
`pairing_optimization/msa_parsing.py` define the same three functions —
`read_msa`, `read_sequence`, `remove_insertions`. A diff shows the only
differences are the header comment and an `__all__` line. The figure scripts
import `msa_parsing` six times and `potts_diffpalm_utils` zero times. Delete
`potts_diffpalm_utils.py`.

**Across the package boundary.** `Cell_2026_analysis_notebooks/Figure5/utils/weights.py`
duplicates the first 253 lines of `proteingym_utils.py`. Both define the same
seven functions. Comparing them by parsing each function and normalising the
syntax tree, **six of the seven are identical** and the seventh,
`map_from_alphabet`, differs only in whether an assertion message uses
`str.format` or an f-string. There is no numerical difference. Section 7 covers
what to do about it.

**`data_downloader.py`** (38 lines) has no importer. It is not a duplicate and
it is not obviously dead — it downloads structures from AlphaFold DB and the
RCSB, which is a plausible thing to call from a notebook. It stays, as
`data/download.py`. Section 6 covers the two functions it calls.

## 3. The organising principle is already in `pyproject.toml`

Do not invent a taxonomy. The repository has one, with reasoning attached, in
its optional dependencies:

```
jacobian    → categorical_jacobian.py         (jax)
proteingym  → proteingym_utils.py             (numba, pandas)
pairing     → pairing_optimization/*          (fair-esm)
typecheck   → custom_typing.py                (beartype)
```

The optional-dependency graph **is** the intended encapsulation boundary. It is
simply not visible in the directory structure. A reader who opens the package
sees 20 flat files and cannot tell which four need extras.

This gives a testable invariant rather than an opinion:

> Importing the model must pull in no optional dependency.

That invariant holds today. Importing `msa_pairformer.model` loads no
`matplotlib`, `sklearn`, `numba`, `pandas`, `jax`, `esm`, `Bio` or `scipy`
module. It holds by accident, not by design, and nothing protects it. A single
`import matplotlib` added to `core.py` for a debugging plot would break it
silently.

A layout whose boundary you can assert in one test is better than one you can
only describe. The proposal makes the tiers into directories so the test can be
written as "no module under `nn/` may import from outside tier 0". Lightning
and torchmetrics arrive as a fifth extra, `training`, under the same rule.

## 4. Proposed layout

```
msa_pairformer/
    __init__.py             MSAPairformer, MSA, aa2tok_d, tok2aa_d,
                            prepare_msa_masks, __version__ — and nothing else

    nn/                     tier 0: no optional dependency, no file I/O
        core.py             unchanged
        model.py            unchanged: CoreModule, MSAPairformer
        heads.py            was regression.py: LMHead,
                            LogisticRegressionContactHead, MRFHead
        pairwise_operations.py, outer_product.py, positional_encoding.py,
        chunk_layer.py, custom_typing.py
                            unchanged

    tokens.py               moved out of dataset.py: aa2tok_d, tok2aa_d,
                            code2aa_d, nTokenTypes, the ESM vocabulary and
                            cross-walk. No Bio, no scipy, no subprocess.
    features.py             moved out of dataset.py: prepare_msa_masks,
                            prepare_inputs, prepare_inputs_bf16, onehot_msa,
                            get_relative_positions, prep_molecule_feats,
                            create_msa_subset_mask, msa_mlm
    msa.py                  moved out of dataset.py: the MSA class, and
                            read_msa / read_sequence / remove_insertions
                            promoted from pairing_optimization/msa_parsing.py

    data/                   file and dataloader plumbing
        msa_datasets.py     MSADataset, CollateAFBatch, trRosetta* — the rest
                            of dataset.py, including the hhfilter subprocess
        download.py         was data_downloader.py
        weighting.py        calc_weights_fast and its numba kernels from
                            proteingym_utils.py, plus
                            fit_seq_weight_mixture_model from utils.py

    training/               Lightning. See section 5.       [training extra]
        datamodule.py       MSADataModule — replaces init_dataloaders
        module.py           MSAPairformerModule — the LightningModule
        metrics.py          torchmetrics collections, if any are custom

    evaluate/               needs matplotlib, sklearn, Bio.PDB
        structure.py        was utils.py, group 1
        contacts.py         was utils.py, group 2
        confind.py          was utils.py, group 3
        coevolution.py      was categorical_jacobian.py     [jacobian extra]
        proteingym.py       was proteingym_utils.py         [proteingym extra]
        plots.py            was plotting.py

    pairing/                was pairing_optimization/       [pairing extra]
        sinkhorn.py         was gumbel_sinkhorn_utils.py
        esm_datasets.py     was pairing_optimization/datasets.py
        mp_em.py, mp_pdp.py, msa_transformer_diffpalm.py
```

Deleted: `potts_diffpalm_utils.py` (duplicate), `dataset.py` (split four ways),
`utils.py` (split three ways, plus two functions retired in section 6),
`pairing_optimization/msa_parsing.py` (promoted to `msa.py`),
`training_utils.py` (rewritten as `training/`, not dropped — section 5).

Five points about the choices.

**Only two modules are renamed inside `nn/`.** `regression.py` becomes
`heads.py`, which is what the file contains. Everything else keeps its name and
only gains the `nn/` prefix. Section 8 explains why the restraint. The
`heads.py` rename costs one edit to `build_cases` in
`tests/generate_fixtures.py`.

**The two `datasets.py` files are renamed apart on purpose.**
`data/msa_datasets.py` and `pairing/esm_datasets.py` would otherwise be two
files with one name, which is the problem of section 2 again at a smaller
scale.

**`tokens.py`, `features.py` and `msa.py` are at the top level, not under
`data/`.** They are model concepts. `data/` becomes what its name says: the
part that touches the disk. This is what makes the "12 of 13 imports" finding
go away.

**Every remaining rename is a `*_utils` file.** Those are the files you asked
about, and after this the package has none.

**File count is not the target.** The layout has 26 files where there were 20.
The measure of success is that a reader can predict which file a thing is in,
and that `nn/` provably imports nothing optional.

## 5. `training/`: what Lightning replaces

`training_utils.py` stays as a capability and goes as a file. It is not dead
code, it is unreleased code — the training loop for a training script this
repository does not yet ship. When that script is written, the 352 lines below
are the wrong starting point, because most of them re-implement what a training
framework provides.

### 5.1 The mapping, class by class

| Today, in `training_utils.py` | Replacement | Why |
| --- | --- | --- |
| `init_dataloaders`, 44 parameters, manual index splitting, returns three loaders | `MSADataModule`: `setup()` plus `train_dataloader()` / `val_dataloader()` / `test_dataloader()` | Most of the 44 are one knob duplicated as `x` and `x_val_test`. A DataModule holds them as fields and splits once in `setup`. |
| `init_trRosetta_contact_dataloaders` | a second `LightningDataModule` | Same shape, different dataset. Two DataModules, one `Trainer`. |
| `GradAccumStatTracker` | nothing. Delete. | Lightning aggregates logged metrics across `accumulate_grad_batches` on its own. This class exists only because the loop is hand-written. |
| `GradAccumLossTracker` | `torchmetrics.text.Perplexity`, `torchmetrics.classification.MulticlassAccuracy` | It keeps `total_loss / total_tokens` and `total_correct / total_tokens`. That is precisely a torchmetrics `update` and `compute`. |
| `LossTracker` (an EMA per category) | `self.log(..., on_step=True)` | Smoothing a curve is the logger's job, not the model's. |
| `evaluate_prediction(logits, batch, device, criterion, ...)` | a private `_shared_step`, called by `training_step` and `validation_step` | It currently takes `device` and `criterion` as arguments. A `LightningModule` owns both. |
| `set_seed` | `lightning.seed_everything(seed, workers=True)` | The current one does not seed dataloader workers. `workers=True` does. |

### 5.2 Two reasons that are not about idiom

**`GradAccumLossTracker` is not DDP-safe.** It sums Python and tensor scalars
into plain attributes and never calls an all-reduce. On one GPU it is correct.
On more than one it silently reports the metrics of rank zero as if they were
the metrics of the run. Any real pre-training run of the size in
[throughput-torch-compile.md](throughput-torch-compile.md) is multi-GPU.
`torchmetrics` synchronises across ranks in `compute`, which is the actual
reason to adopt it.

**`set_seed` sets `torch.backends.cudnn.deterministic = True` and
`benchmark = False`.** That is a throughput cost, hard-coded into a helper, in
a repository whose recent work is entirely throughput measurement. Under
Lightning the two halves separate: `seed_everything` handles reproducibility of
the data and the initialisation, and `Trainer(deterministic=...)` makes the
kernel-selection cost an explicit per-run decision. The current code makes it
for you and does not say so.

### 5.3 The constraint that Lightning must not break

`bench/step.py` compiles five submodules in place with `Module.compile(mode=...)`,
and `test_compile_targets_cover_every_repeated_leaf` asserts it finds 22 × 4 of
them. Lightning's `Trainer` has its own handling of compilation, of mixed
precision, and of gradient accumulation. Those overlap with the harness.

So: **the benchmark harness keeps driving the model directly. It must not go
through `Trainer`.** This is free if `training/` is additive — the
`LightningModule` wraps the same `MSAPairformer` that `bench/` already builds,
and `bench/` never imports `training/`. It stops being free the moment
something in `nn/` starts assuming a `Trainer` is present. Nothing in the
proposal does, and nothing should.

### 5.4 Dependencies

`lightning` and `torchmetrics` are in neither `pyproject.toml` nor `uv.lock`
today. They go in a new `training` extra, not in the base dependencies. The
invariant in section 3 must survive: installing the model to run inference must
not install a training framework.

## 6. Why `_run` and `write_log` exist, and what replaces them

You asked why these are hand-rolled instead of idiomatic. Here is `_run`:

```python
def _run(x):
    "Generic run."
    if isinstance(x, str):
        res = subprocess.run(x.split(' '), capture_output=True, text=True)
    elif isinstance(x, list):
        res = subprocess.run(x, capture_output=True, text=True)
    else:
        print("Must pass a string or a list of strings to _run()")
        return -1
    res.stdout = res.stdout.strip().split('\n')
    res.stderr = res.stderr.strip().split('\n')
    return res
```

It has five defects. Two of them lose information, which is the answer to the
question:

1. **It never checks `returncode`.** `subprocess.run(..., check=True)` raises
   `CalledProcessError` on a non-zero exit. Without it, a failed `wget` in
   `download_AF_pdb` writes an error to a log file and returns `None`. The
   caller cannot tell a download that failed from one that worked.
2. **It rewrites `res.stdout` and `res.stderr` into lists of lines.** The
   returned object is still a `CompletedProcess`, so it claims a contract it no
   longer honours. Anything that reads `.stdout` expecting a string gets a
   `list`. Empty output becomes `['']`, not `[]`.

The other three are ordinary:

3. `x.split(' ')` is not shell tokenisation. Any path with a space in it breaks.
   `shlex.split` is the standard-library answer.
4. Bad input prints and returns `-1`, so the failure is a sentinel a caller must
   remember to test, not a `TypeError`.
5. `write_log` appends to `fout + '.out'` and `fout + '.err'` forever, with no
   rotation. `data_downloader.py` passes `download.log` as the prefix, so the
   files it actually produces are `download.log.out` and `download.log.err`.

The replacement is one line plus the standard `logging` module:

```python
res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
logger.debug("%s", res.stdout)
```

A library should log to a logger and let the application decide where the
output goes. Writing two files next to the data is a policy, and `utils.py` is
not the place that policy belongs.

**Why they exist** — this is inference, not evidence: the code has the shape of
notebook-lineage work. Line-lists are convenient at a REPL, printing instead of
raising keeps a cell alive, and a log file beside the data is how you debug a
batch of twenty thousand downloads with no logging configured. All reasonable at
the bench. The tell that it outgrew that is `_run` being private and imported
across modules anyway.

Both functions are retired rather than moved. With `check=True` and a logger at
the two call sites — `data/download.py` and `evaluate/confind.py` — nothing is
left for a shared helper module to hold.

**One further observation.** `data_downloader.py` shells out to `wget` to
perform an HTTP GET. `wget` is an external binary declared nowhere in
`pyproject.toml`, and it is absent by default on macOS. `urllib.request` or
`httpx` removes the dependency and returns a status code the caller can act on.
That is a behaviour change, so it is question 3 in section 10 rather than part
of the layout.

## 7. `Cell_2026_analysis_notebooks/Figure5/`

Five of the ten `utils` are here. Two changes remove them. Keep the changes in
separate commits: if a figure result moves, you want to know which one did it.

### 7.1 Make it a package. This is a bug fix.

`Figure5B_ProteinGym_VEP_ProteinGym_MSAs.py` does this:

```python
sys.path.append("utils")
from compute_fitness_utils import *
```

and inside that directory the modules import each other by bare top-level name:
`import msa_utils`, `from weights import map_from_alphabet, map_matrix,
calc_weights_fast`, `from scoring_utils import get_mutated_sequence`. There is
no `__init__.py`.

**The consequence is that the figure script only runs when the working directory
is `Figure5/`.** Run it from the repository root and the `sys.path.append`
points at a directory that does not exist, and the import fails. That is not a
style problem. It is the reason a reproduction attempt fails on the first try.

Add `__init__.py`, rename the directory to something that is not `utils`, and
convert the bare imports to relative ones. `compute_fitness_utils.py` becomes
`fitness.py`, `msa_utils.py` becomes `msa_io.py`, `scoring_utils.py` becomes
`scoring.py`, `data_utils.py` becomes `datasets.py`.

One caveat on the star import. `from compute_fitness_utils import *` also
re-exports whatever that module imported, including names from `msa_utils`. The
figure script may use names it never mentions. Converting to explicit imports
therefore needs the script run once, not only a lint pass.

### 7.2 Delete `weights.py` and import from the package

`weights.py` and the first 253 lines of `proteingym_utils.py` define the same
seven functions. The comparison method matters, so state it: each function was
parsed, its syntax tree normalised by round-tripping through `ast.unparse`, and
the results compared. **Six of the seven are identical.** The seventh,
`map_from_alphabet`, differs only in whether an assertion message is built with
`str.format` or an f-string.

So the figure directory can import `calc_weights_fast`, `map_from_alphabet` and
`map_matrix` from `msa_pairformer.data.weighting` and the file can go. The
numbers do not change. Sequence weighting drops from three homes to one.

### 7.3 What stays frozen

The figure scripts keep their names. `Figure3EFG_MP_PDP_HK_RR.py` says which
panel it produces, and that is the whole value of the directory. Only their
import lines change, and only where section 4 moved the target.

## 8. What any layout must not break

Four constraints. Each is a real mechanism in this repository, not a style
preference. They are the reason section 4 renames so little.

**1. The `MSA_Pairformer.py` shim — resolved, and it is deleted.** The shim
promises that `from MSA_Pairformer.model import MSAPairformer` keeps working for
code written against release 1.0.2. Its finder is a pure string operation —
`find_spec` tests `fullname.startswith("MSA_Pairformer.")` and asks for
`"msa_pairformer." + fullname[len(prefix):]`. There is no table of module names
in it, so a re-export stub at the old path would have been the only way to keep
the promise across a move.

There are no 1.0.2 consumers to keep the promise for. So phase 4 deletes
`MSA_Pairformer.py` and its `force-include` entry in `pyproject.toml`, and this
constraint goes with it. **No re-export stubs are written anywhere in this
refactor.** Every call site is updated in the same commit that moves its target.

This is independent of `tests/generate_fixtures.py`'s `pkg` argument, which
points at a separate upstream *worktree* and keeps working.

**2. `bench/step.py` matches five concrete classes by `isinstance`.**
`COMPILE_TARGETS` is `(MSAPairWeightedAveraging, OuterProduct, PairwiseBlock,
PreLayerNorm, Transition)`, and `test_compile_targets_cover_every_repeated_leaf`
asserts that selection finds 22 × 4 per-layer targets. Those five classes must
stay importable and must stay the same module kinds. If they do not, the
`torch.compile` measurement in
[throughput-torch-compile.md](throughput-torch-compile.md) stops measuring what
it claims. Section 5.3 adds the Lightning half of this constraint.

**3. `tests/generate_fixtures.py` builds module paths as strings.**
`build_cases` calls `importlib.import_module(f"{pkg}.core")` and the same for
`pairwise_operations`, `outer_product`, `positional_encoding`, `regression` and
`model`. The `pkg` argument exists so the same script can run against the
*upstream* tree under its old name. So those six paths must resolve in both
trees. Either keep the six module names reachable at their current paths, or
give `build_cases` a per-tree path map. The `regression.py` → `heads.py` rename
makes the second option necessary.

The goldens themselves survive a move. Cases are keyed by case name, and
`param_fingerprint` records `state_dict` key names and per-key statistics — not
module paths or qualified names. **No re-fixturing is needed.** Only
`build_cases` needs editing.

**4. `bench/modal_app.py:149`** calls
`add_local_python_source("msa_pairformer", "bench")`. Top-level names only, so
internal moves are safe. A package rename is not.

## 9. Migration

Six commits. Each is independently revertable and each leaves the test suite
green.

**Commit 1 — the facade. Additive; breaks nothing.**
Write `__init__.py` with `MSAPairformer`, `MSA`, `aa2tok_d`, `tok2aa_d`,
`prepare_msa_masks` and `__version__`. Add `__all__`. Do not import
`matplotlib`, `sklearn` or any extra at package import time; that would destroy
the invariant in section 3 for every user at once. Add the test that asserts the
invariant. This commit alone gives most of the benefit, because from here
onwards a new consumer has a shallow path to import, and `training/` in commit 5
is exactly such a consumer.

**Commit 2 — delete the in-package duplicate.**
Remove `potts_diffpalm_utils.py`. It duplicates a file the notebooks already
import and it has no caller. One file, no decisions.

**Commit 3 — split the two bags; retire `_run` and `write_log`.**
`dataset.py` becomes `tokens.py`, `features.py`, `msa.py` and
`data/msa_datasets.py`. `utils.py` becomes `evaluate/structure.py`,
`evaluate/contacts.py` and `evaluate/confind.py`, and its sequence-weighting
orphan joins `data/weighting.py`. `data_downloader.py` becomes
`data/download.py`. Both `data/` and `evaluate/` are created here, holding only
what this commit splits — so no file moves twice. Convert the four `_run` call
sites in `data/download.py` and the one in `evaluate/confind.py` to
`subprocess.run(shlex.split(cmd), ..., check=True)` with a module logger, and
delete both helpers. Update every call site in `bench/`, `tests/` and the figure
scripts in this same commit.

**Commit 4 — the remaining directories.**
Move the tier-0 modules into `nn/` and rename `regression.py` to `nn/heads.py`.
Move `categorical_jacobian.py`, `proteingym_utils.py` and `plotting.py` into the
`evaluate/` that commit 3 created. Rename `pairing_optimization/` to `pairing/`,
with `gumbel_sinkhorn_utils.py` becoming `sinkhorn.py` and `datasets.py`
becoming `esm_datasets.py`. Delete `MSA_Pairformer.py` and its `force-include`
in `pyproject.toml`. Update `build_cases` in `tests/generate_fixtures.py`: new
paths for the current tree, old paths for the upstream tree. Re-run
`bench/drift.py` to confirm the goldens still match — not because a move should
change numerics, but because constraint 3 is the one thing here that touches the
correctness suite.

**Commit 5 — `training/`.**
Add the `training` extra. Write `MSADataModule` and `MSAPairformerModule`
against the facade from commit 1. Delete `training_utils.py` in the same commit
that replaces it, so the two never diverge. `bench/` is untouched, by section
5.3.

**Commit 6 — the figure directory.**
Section 7.1 then section 7.2, as two commits if you prefer. Run
`Figure5B_ProteinGym_VEP_ProteinGym_MSAs.py` from the repository root
afterwards, which is the check that 7.1 actually fixed what it claims.

## 10. Decisions taken

Four questions were open when this document was first written. All are now
answered, and the answers are already applied above.

1. **No external consumers.** All names are available for refactoring. This
   removes the re-export stubs from commits 3 and 4, removes constraint 1 from
   section 8, and deletes `MSA_Pairformer.py` in commit 4.
2. **No Lightning logger yet.** `training/` calls `self.log(...)` and configures
   no logger. A CSV or Weights and Biases logger is a later, separate change,
   and it is a `Trainer` argument, not a change to `training/`.
3. **`wget` stays.** Shelling out to it is out of scope. The `check=True`
   conversion in commit 3 still applies, because that is about not discarding
   the exit status, not about which binary runs.
4. **`training_utils.py` is unreleased, not dead.** It is rewritten as
   `training/`, per section 5.

## 11. How each phase is verified

Behaviour must not change. Three gates run after every commit.

**The test suite.** `pytest` gives 41 passed and 33 skipped at commit `ddcbc26`,
and 44 passed with `MSA_PAIRFORMER_TYPECHECK=1`. Both must hold. Twenty-six of
the skips are CUDA variants of the golden replay; the CPU replay does run, and
it covers `MSAPairformer.forward`, `predict_cb_contacts` and
`predict_confind_contacts` against fixtures recorded from the upstream
implementation. That is the numerics gate.

**A symbol-level snapshot.** Every function and class in the package is parsed
with `ast`, normalised by a round trip through `ast.unparse`, and hashed. The
result is keyed by **symbol name, not module path**, so a pure move is invisible
and only a real edit registers. It reads source off disk rather than importing,
which is what lets it cover `categorical_jacobian`, `proteingym_utils` and
`pairing_optimization/` — none of which can be imported without extras that are
not installed. The tokenizer dictionaries are compared by value as well as by
source, because commit 3 hand-carries them into a new file. The baseline holds
248 symbols and 8 constants.

**`ruff` and `ty`.** The gate is `uvx ruff check bench tests` and
`uvx ty check bench tests`, both of which pass at `ddcbc26` and must keep
passing. It is **not** the whole package: `uvx ruff check msa_pairformer` finds
64 errors at `ddcbc26` — 21 `B006` mutable default arguments, 10 `E741`
ambiguous names, 9 `F841` unused locals, and 24 others — and has never been
clean. That legacy is out of scope here; fixing it would be a behaviour change
smuggled into a move.

The count is pinned instead. Moving a function carries its lint errors with it,
so the package total must stay at exactly 64 through commits 2, 3 and 4. A
higher number means new code was written where only moved code was expected.
Any file created by this refactor must itself be clean.

Count the diagnostics, not the output lines:

```
uvx ruff check msa_pairformer --output-format=concise | grep -c '^msa_pairformer/'
```

`wc -l` gives 66, because ruff also prints a total and a count of what is
auto-fixable. The second of those two lines appears only when something is
fixable, so it can come and go on its own and move the total independently of
the code.

Commits 2, 3 (the moves) and 4 must show **no symbol-level change at all**.
Commits 3 (the `_run` retirement), 5 and 6.2 change source on purpose, so for
those the expected diff is written down first and the verifier checks the actual
diff matches it. For commit 6.2 the expectation is already known: the seven
functions disappear from the figure directory and the copies in
`msa_pairformer` are untouched.
