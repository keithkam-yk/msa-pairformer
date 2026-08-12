# MSA Pairformer
<!-- ![Logo](msa_pairformer_logo.png) -->
<div align="left">
  <img src="msa_pairformer_logo.png" width="300" alt="Neural Network Logo">
</div>

[![Contact prediction](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoakiyama/MSA_Pairformer/blob/main/MSA_Pairformer_with_MMseqs2.ipynb/)

- [MSA Pairformer](#MSA-Pairformer)
  - [Installation ](#installation-)
  - [MSA Pairformer](#MSA-Pairformer--)
    - [Quickstart ](#quickstart--)
    - [Example Usage](#example-usage--)
  - [Licenses  ](#licenses--)
  - [Citations  ](#citations--)

This repository contains the latest release of MSA Pairformer and Google Colab notebooks for relevant analyses. Here, you will find how to use MSA Pairformer to embed protein sequences, predict residue-residue interactions in monomers and at the interface of protein-protein interactions, and perform zero-shot variant effect prediction.

## Installation <a name="installation"></a>

To get started with MSA Pairformer, install the python library using pip:

```bash
pip install msa-pairformer
```

or download this Github repository and install manually
```bash
git clone git@github.com:yoakiyama/MSA_Pairformer.git
pip install -e .
```

### Developing on the repository

Dependencies are locked with [uv](https://docs.astral.sh/uv/). One command creates
the virtualenv, installs the project in editable mode, and installs the dev tools:

```bash
uv sync
```

That resolves from `uv.lock`, so everyone gets the same versions — which matters
more than usual here: the golden fixtures in `tests/fixtures/` were recorded under
a specific torch build, and a different one shifts the numbers the correctness
suite compares against.

Run things through `uv run` so they use that environment:

```bash
uv run pytest
```

Linting and type checking are deliberately not in the lock — they run as one-off
tools, which keeps their versions out of the runtime environment:

```bash
uvx ruff check
uvx ty check
```

The optional extras (`jacobian`, `proteingym`, `pairing`, `analysis`, `typecheck`)
are for installing users and are not synced by default. Add them as needed:

```bash
uv sync --extra analysis
```

### Training throughput

[docs/throughput-baseline.md](docs/throughput-baseline.md) records the first
throughput baseline: what the harness in `bench/` measures, what it excludes, the
memory ceiling on an 80 GB H100, and the extrapolation to the paper's MSA depth.

### Upgrading from 1.0.2 and earlier

The python package was renamed from `MSA_Pairformer` to `msa_pairformer` (PEP 8). Import
it under the new name:

```py
from msa_pairformer.model import MSAPairformer     # new
from MSA_Pairformer.model import MSAPairformer     # old, deprecated but still works
```

The old spelling keeps working through a compatibility shim (`MSA_Pairformer.py`) that
aliases `MSA_Pairformer.*` onto the identical `msa_pairformer.*` modules and raises a
`DeprecationWarning`. Both spellings resolve to the same module objects, so mixing them
is safe. The shim will be removed in a future release -- please migrate your imports.

Note that `MSA_Pairformer/` and `msa_pairformer/` are the *same* directory on
case-insensitive filesystems (macOS, Windows). If you are upgrading in place there,
`pip uninstall msa-pairformer` before installing the new version so that no files from
the old package directory are left behind.

### Installing hhsuite (for filtering MSAs)

We use hhfilter (part of hhsuite) as the default option for subsampling MSAs. To install hhsuite, please follow these directions:

```
curl -fsSL https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz
tar xz -C hhsuite
export PATH="$PATH:/path/to/hhsuite/bin:/path/to/hhsuite/scripts"
```
Alternatively, in python:
```
def _setup_tools():
  """Download and compile C++ tools."""

  # Install HHsuite
  hhsuite_path = "hhsuite"
  if not os.path.isdir(hhsuite_path):
      print("Installing HHsuite...")
      os.makedirs(hhsuite_path, exist_ok=True)
      url = "https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz"
      os.system(f"curl -fsSL {url} | tar xz -C {hhsuite_path}/")
  os.environ['PATH'] += f":{hhsuite_path}/bin:{hhsuite_path}/scripts"
```

## MSA Pairformer <a name="MSA-Pairformer"></a>

[MSA Pairformer](https://www.cell.com/cell/fulltext/S0092-8674(26)00749-X) is an MSA-based protein language model that can model the coevolution of interacting proteins. In this repository, we provide the model source code, a Google Colab notebook for quick experimentation, and notebooks and scripts to reproduce the results of our manuscript. We are excited to deliver this tool to the community and to see all of its applications as a tool to study and engineer biology.

### Getting started with MSA Pairformer <a name="getting-started"></a>
The model's weights can be downloaded from Huggingface under [HuggingFace/yakiyama/MSA-Pairformer](https://huggingface.co/yakiyama/MSA-Pairformer/).
```py
import torch
import numpy as np
from huggingface_hub import login
from msa_pairformer.model import MSAPairformer
from msa_pairformer.dataset import MSA, prepare_msa_masks, aa2tok_d

# Use the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {torch.cuda.get_device_name(device)}")

# This function will allow you to login to huggingface via an API key
login()

# Download model weights and load model
# As long as the cache doesn't get cleared, you won't need to re-download the weights whenever you re-run this
model = MSAPairformer.from_pretrained(device=device)

# You can also save the downloaded weights to a specified directory in your filesystem.
# Saving the model weights like so will allow you to load the model without re-downloading if your cache gets cleared.
# Once you run this code once, you can re-run and it will automatically load the weights
save_model_dir = "model_weights"
model = MSAPairformer.from_pretrained(weights_dir=save_model_dir, device=device)
```

Here, we walk through how to run MSA Pairformer on the phenylalanyl tRNA synthetase pheS and pheT dimer (PDB ID: 1B70).
```py
# Subsample MSA using hhfilter and greedy diversification
msa_file = "data/1B70_A_1B70_B.fas"
max_msa_depth = 512
max_length = 10240
chain_break_idx = 265
np.random.seed(42)
msa_obj = MSA(
    msa_file_path=msa_file,
    max_seqs=max_msa_depth,
    max_length=max_length,
    max_tokens=np.inf,
    diverse_select_method="hhfilter",
    hhfilter_kwargs={"binary": "hhfilter"}
)
# Prepare MSA and mask tensors
msa_tokenized_t = msa_obj.diverse_tokenized_msa
msa_onehot_t = torch.nn.functional.one_hot(msa_tokenized_t, num_classes=len(aa2tok_d)).unsqueeze(0).float().to(device)
mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(msa_obj.diverse_tokenized_msa.unsqueeze(0))
mask, msa_mask, full_mask, pairwise_mask = mask.to(device), msa_mask.to(device), full_mask.to(device), pairwise_mask.to(device)

# Run MSA Pairformer to generate embeddings and predict contacts
with torch.no_grad():
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        res = model(
            msa=msa_onehot_t.to(torch.bfloat16),
            mask=mask,
            msa_mask=msa_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            complex_chain_break_indices=[[chain_break_idx]],
            return_seq_weights=True,
            return_pairwise_repr_layer_idx=None,
            return_msa_repr_layer_idx=None,
            return_cb_contacts=True,
            return_confind_contacts=True
        )
# res is a dictionary with the following keys: final_msa_repr, final_pairwise_repr, msa_repr_d, pairwise_repr_d, seq_weights_list_d, predicted_cb_contacts, predicted_confind_contacts

# Just predict Cb-Cb
with torch.no_grad():
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        res = model.predict_cb_contacts(
            msa=msa_onehot_t.to(torch.bfloat16),
            mask=mask,
            msa_mask=msa_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            complex_chain_break_indices=[[chain_break_idx]],
            return_seq_weights=True,
        )
# Just predict ConFind contacts
with torch.no_grad():
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        res = model.predict_confind_contacts(
            msa=msa_onehot_t.to(torch.bfloat16),
            mask=mask,
            msa_mask=msa_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            complex_chain_break_indices=[[chain_break_idx]],
            return_seq_weights=True,
        )
```

That's it -- you've generated embeddings and predicted contacts using MSA Pairformer!

## Licenses <a name="licenses"></a>

MSA Pairformer code and model weights are released under a permissive, slightly modified ☕️ MIT license. It can be freely used for both academic and commercial purposes.

## Citation <a name="citation"></a>
If you use MSA Pairformer in your work, please use the following citation
```
@article {Akiyama2026,
	author = {Akiyama, Yo and Zhang, Zhidian and Tang, Olivia and Kim, Rachel Seongeun and Mirdita, Milot and Steinegger, Martin and Ovchinnikov, Sergey},
	title = {Expanding the scope of protein language modeling to protein-protein interactions with MSA Pairformer},
	year = {2026},
	doi = {10.1016/j.cell.2026.06.029},
	publisher = {Cell Press},
	URL = {https://doi.org/10.1016/j.cell.2026.06.029},
	journal = {Cell}
}

```
