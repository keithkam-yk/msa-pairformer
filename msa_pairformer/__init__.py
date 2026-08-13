"""The public surface of the package.

Until now this file was empty, which published all 20 modules: a consumer had to
write `from msa_pairformer.tokens import aa2tok_d`, so every internal file name
became API and nothing could move. The names below are the ones a consumer
actually needs -- load the model, read an alignment off disk, tokenise it, build
the masks the forward pass wants -- and they are the only ones the package
promises to keep at a stable path. Everything else is an implementation detail.

**Every export resolves lazily, through PEP 562, and that is load-bearing rather
than a micro-optimisation.** Importing a submodule imports its parent package
first, so an eager `from msa_pairformer.msa import MSA` here would be
charged to `import msa_pairformer.nn.model` -- a path that pays for none of it
today. Measured on this tree:

    import msa_pairformer.nn.model  torch, huggingface_hub. No Bio, no scipy.
    import msa_pairformer.msa       torch, Bio, scipy. No huggingface_hub.

Those two sets are disjoint in both directions, which is why `MSAPairformer` is
deferred alongside the rest: eager, it would add the model stack to every
consumer that only wanted a tokenizer, exactly as an eager `MSA` would add
`Bio.SeqIO` to every consumer that only wanted the model. A facade is supposed
to be free to import. `bench/`, `tests/` and the figure scripts all reach for
deep paths, and none of them should start paying for a neighbour.

This is not scaffolding for the module split either. Now that `dataset.py` has
become `tokens.py`, `features.py` and `msa.py`, `MSA` still sits behind `Bio`,
so the laziness stays earned.

`tests/test_imports.py` holds the line, at runtime and by source inspection.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Never executed: a type checker reads these bindings, and the interpreter
    # skips the block. It is what keeps the lazy names resolvable for `ty` and
    # for editor completion without reintroducing the import cost.
    from msa_pairformer.features import prepare_msa_masks
    from msa_pairformer.msa import MSA
    from msa_pairformer.nn.model import MSAPairformer
    from msa_pairformer.tokens import aa2tok_d, tok2aa_d

# Export -> the module that defines it. The mapping is data rather than a chain
# of imports inside `__getattr__` so that `__dir__` can answer from the same
# source, and so the phase that moved `MSA` into `msa.py` edited one string.
_EXPORTS = {
    "MSA": "msa_pairformer.msa",
    "MSAPairformer": "msa_pairformer.nn.model",
    "aa2tok_d": "msa_pairformer.tokens",
    "prepare_msa_masks": "msa_pairformer.features",
    "tok2aa_d": "msa_pairformer.tokens",
}

try:
    # Read from the installed distribution instead of repeating the number here,
    # so `__version__` and `pyproject.toml` cannot drift apart. It also reports
    # what the consumer really has installed, which is the answer that is useful
    # in a bug report -- a literal would report what the checkout claims. This
    # one stays eager: `importlib.metadata` is standard library, and a version
    # string is the one thing a caller may want without loading torch at all.
    __version__ = version("msa-pairformer")
except PackageNotFoundError:
    # Reached only when the package was never installed: running straight out of
    # a clone with the repository root on `sys.path`. Keep in step with
    # `pyproject.toml`'s `version`.
    __version__ = "1.0.2"

__all__ = [
    "MSA",
    "MSAPairformer",
    "__version__",
    "aa2tok_d",
    "prepare_msa_masks",
    "tok2aa_d",
]


def __getattr__(name: str) -> object:
    """Resolve an export on first use, then bind it as a real module global.

    The write into `globals()` matters: after one access the name is found by
    normal attribute lookup and this function is never consulted for it again,
    so the deferral costs one dictionary miss per name per process rather than
    an indirection on every use.

    An unknown name raises `AttributeError` with the interpreter's own wording,
    so `hasattr`, `getattr` with a default, and a typo at the REPL all behave as
    they would on any other module.
    """
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Defining `__getattr__` hides the lazy names from `dir()`, so put them back.

    Without this, tab-completion on the package would list `__version__` and
    nothing else until something happened to touch an export first -- the
    completion list would depend on import history.

    `globals()` stays in the union rather than being replaced by the facade:
    that is what a module without a `__getattr__` reports, and it is what keeps
    `__name__`, `__spec__` and any already-imported submodule visible to
    `inspect.getmembers`. Narrowing to the six exports would read as tidier and
    would be a behaviour change for no gain.
    """
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS))
