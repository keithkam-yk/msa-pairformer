"""The facade costs no dependency the deep paths did not already pay for.

The optional-dependency graph in `pyproject.toml` is this package's real
encapsulation boundary, and it yields an invariant that can be checked rather
than described:

    Importing the model must pull in no optional dependency.

It held by accident before this file existed, and it is fragile in a way that
is easy to miss: importing a submodule imports its parent package first, so
anything `msa_pairformer/__init__.py` imports is charged to
`import msa_pairformer.model` as well. A facade written the obvious way would
therefore have made every consumer of a deep path -- `bench/`, `tests/`, the
Cell 2026 figure scripts -- pay for `Bio.SeqIO` and `scipy` to load a model,
and every consumer that only wanted a tokenizer pay for `huggingface_hub`. The
facade defers its exports for exactly this reason; these tests are what stops
the deferral from being quietly undone.

Two checks, and neither is sufficient alone:

* The runtime check imports in a fresh interpreter and looks at `sys.modules`,
  so it sees the whole transitive graph however a module was reached --
  including through `importlib.import_module`, which reads a string no parser
  can follow. But it can only observe what is installed. `numba`, `pandas`,
  `jax` and `esm` are absent from this environment, so a suppressed import
  (`try: import numba / except ImportError`) or a lazy path passes here
  vacuously and then pulls the dependency in on a machine that has it.
* The static check parses the tier-0 sources and never imports anything, so it
  answers independently of what is installed -- catching precisely that case.
  But it only sees the files it enumerates, and only literal import statements.

A third test keeps the second one's file list honest, because a guard that
names its subjects by hand covers a new module only if someone remembers to
add it.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import msa_pairformer

# Not unused, and not redundant with the line above: binding the submodule is
# what sets `model` as an attribute of the package, which is the baseline
# `dir()` behaviour the last test asserts `__dir__` did not narrow away.
import msa_pairformer.model

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "msa_pairformer"

# Behind an extra in pyproject.toml -- jacobian -> jax, proteingym ->
# numba/pandas, pairing -> esm -- or, for matplotlib and sklearn, declared as a
# base dependency today but reached only by the analysis code that later phases
# move under `evaluate/`. Importing one of these can fail outright on a machine
# that installed the model and nothing else.
#
# `beartype` is deliberately absent. The typecheck extra is the one the suite
# switches on itself, via MSA_PAIRFORMER_TYPECHECK=1; the child process below
# inherits that environment and `custom_typing.py` imports beartype when it is
# set. On the list, this test would pass or fail depending on how pytest was
# invoked.
OPTIONAL_DEPENDENCIES = ("matplotlib", "sklearn", "numba", "pandas", "jax", "esm")

# Base dependencies, so these are always installed and importing them cannot
# fail. They are excluded for cost on the model path, not for installability:
# they are what `dataset.py` pulls in, and charging them to
# `import msa_pairformer.model` is the regression this file exists to pin. Kept
# in a separate group so a failure says which of the two arguments it broke.
UNWANTED_BASE_DEPENDENCIES = ("Bio", "scipy")

FORBIDDEN = OPTIONAL_DEPENDENCIES + UNWANTED_BASE_DEPENDENCIES

# Each import path a consumer writes today, against what must not appear in its
# graph. The facade is the new path; the two deep ones are what every existing
# caller in `bench/`, `tests/` and the figure scripts writes.
#
# The `dataset.py` entry pins the other direction of the same claim, and it
# needs its own list: `Bio` and `scipy` are that module's own dependencies and
# are expected there. `huggingface_hub` is the marker in that direction --
# `model.py` imports it and `dataset.py` does not, so its presence would mean
# the facade had resolved `MSAPairformer` eagerly and charged the whole model
# stack to a caller that wanted a tokenizer.
IMPORT_PATHS = {
    "msa_pairformer": FORBIDDEN,
    "msa_pairformer.model": FORBIDDEN,
    "msa_pairformer.dataset": (*OPTIONAL_DEPENDENCIES, "huggingface_hub"),
}

# Tier 0 of the target layout (§4): the modules that become `nn/`, plus the
# facade itself. These are the files on the inference path, and the ones the
# invariant is about.
TIER_0_MODULES = (
    "__init__.py",
    "core.py",
    "model.py",
    "regression.py",
    "pairwise_operations.py",
    "outer_product.py",
    "positional_encoding.py",
    "chunk_layer.py",
    "custom_typing.py",
)

# Everything else in the package, with the reason it is not tier 0. This exists
# so the classification can be checked for completeness rather than trusted: a
# module added to the package and forgotten here fails the test below, which is
# the only way a new tier-0 file gets covered from birth. Phase 4 creates
# `nn/__init__.py` -- exactly the kind of file where a convenience import
# appears -- and a hand-maintained list would not have seen it.
NOT_TIER_0 = {
    "dataset.py": "reads alignments from disk: Bio.SeqIO, scipy.spatial, an hhfilter subprocess",
    "utils.py": "structure I/O, contact metrics, CONFIND wrappers: Bio.PDB, sklearn",
    "plotting.py": "matplotlib",
    "categorical_jacobian.py": "jacobian extra: jax",
    "proteingym_utils.py": "proteingym extra: numba, pandas",
    "data_downloader.py": "shells out to wget to fetch structures",
    "training_utils.py": "the hand-written training loop; phase 5 replaces it with training/",
}

# The pairing extra (fair-esm) in its entirety, excluded as a directory because
# its six modules share one reason and none of them is reachable from the model.
NOT_TIER_0_PACKAGES = ("pairing_optimization",)


def probe(module: str, forbidden: tuple[str, ...]) -> str:
    """A one-liner that imports `module` and reports which forbidden names loaded."""
    return (
        "import json, sys\n"
        f"import {module}\n"
        f"print(json.dumps([name for name in {list(forbidden)!r} if name in sys.modules]))\n"
    )


def is_type_checking_guard(node: ast.If) -> bool:
    """Match both `if TYPE_CHECKING:` and `if typing.TYPE_CHECKING:`."""
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def absolute_imports(node: ast.AST):
    """Every absolute import reachable at runtime, as `(top-level name, line)` pairs.

    Relative imports are skipped: `from .core import ...` cannot reach outside
    the package, so it can never be a forbidden dependency. Imports nested in a
    function count the same as module-level ones -- a tier-0 module has no
    business naming these at all, and deferring one only moves the cost to
    whoever calls the function.

    `if TYPE_CHECKING:` bodies are exempt, and that is the one exemption. The
    invariant is about what an import costs at runtime, and those statements
    never execute -- the facade itself uses the pattern to keep its lazy names
    resolvable for `ty` and for editor completion. The `else:` branch of such a
    guard does run, so it is still walked.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name.split(".")[0], node.lineno
        return
    if isinstance(node, ast.ImportFrom):
        if node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno
        return

    children = ast.iter_child_nodes(node)
    if isinstance(node, ast.If) and is_type_checking_guard(node):
        children = iter(node.orelse)
    for child in children:
        yield from absolute_imports(child)


def package_modules() -> list[str]:
    """Every `.py` in the package, excluding the wholly-excluded subpackages."""
    found = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in NOT_TIER_0_PACKAGES:
            continue
        found.append(relative.as_posix())
    return found


@pytest.mark.parametrize(("module", "forbidden"), sorted(IMPORT_PATHS.items()))
def test_import_pulls_in_no_forbidden_dependency(module, forbidden):
    """Checked in a fresh interpreter, because `sys.modules` is session-wide.

    In-process the check would answer a different question: what the whole
    pytest session has imported -- plugins, other test modules, anything torch
    reached for -- rather than what one import statement costs on its own. And
    by the time any test runs, all three of these are long since imported, so
    there would be nothing left to observe.
    """
    # cwd rather than the installed distribution, so the working tree is what
    # gets imported even in a checkout with no editable install.
    proc = subprocess.run(
        [sys.executable, "-c", probe(module, forbidden)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"`import {module}` failed in a fresh interpreter:\n{proc.stderr}"

    pulled_in = json.loads(proc.stdout)
    assert not pulled_in, (
        f"`import {module}` pulled in: {', '.join(pulled_in)}. "
        "Every consumer of this path now pays for them at import time."
    )


def test_tier_0_sources_name_no_forbidden_dependency():
    """The half of the invariant that does not depend on what is installed."""
    violations = [
        f"{name} imports {imported} (line {line})"
        for name in TIER_0_MODULES
        for imported, line in absolute_imports(ast.parse((PACKAGE_ROOT / name).read_text()))
        if imported in FORBIDDEN
    ]
    assert not violations, "tier-0 modules must not import these:\n" + "\n".join(violations)


def test_every_module_is_classified_as_tier_0_or_not():
    """A guard that names its subjects is only as good as the naming.

    Renaming a tier-0 module already fails loudly, because the test above opens
    it by path. Adding one is the silent case: the new file is simply not
    checked, and nothing says so.
    """
    unclassified = [name for name in package_modules() if name not in TIER_0_MODULES and name not in NOT_TIER_0]
    assert not unclassified, (
        f"new module(s) in the package, not classified: {', '.join(unclassified)}. "
        "Add to TIER_0_MODULES to put them under the import guard, or to NOT_TIER_0 with the reason."
    )

    # The other direction: a classified module that no longer exists means a
    # move happened and one of the two lists was not updated with it.
    known = set(TIER_0_MODULES) | set(NOT_TIER_0)
    assert not (known - set(package_modules())), f"classified but missing: {', '.join(sorted(known - set(package_modules())))}"


def test_all_is_the_documented_facade_and_every_name_resolves():
    """`__all__` is a promise on both sides: nothing missing, nothing extra."""
    missing = [name for name in msa_pairformer.__all__ if not hasattr(msa_pairformer, name)]
    assert not missing, f"listed in __all__ but not importable from the package: {', '.join(missing)}"

    # The facade is small on purpose -- a name added here is a name that can no
    # longer move -- so the set is pinned rather than left to grow by habit.
    assert set(msa_pairformer.__all__) == {
        "MSA",
        "MSAPairformer",
        "__version__",
        "aa2tok_d",
        "prepare_msa_masks",
        "tok2aa_d",
    }

    # Defining `__getattr__` costs the exports their place in `dir()` unless
    # `__dir__` puts them back -- but it must put them back *on top of* what a
    # plain module reports, not instead of it. An equality check rather than a
    # subset check, because a subset check passes for either behaviour and this
    # is the one that regressed: submodules and the module dunders disappeared
    # from `inspect.getmembers` when `__dir__` answered with the facade alone.
    assert set(dir(msa_pairformer)) == set(vars(msa_pairformer)) | set(msa_pairformer.__all__)
    assert {"__name__", "__spec__", "__all__", "model"} <= set(dir(msa_pairformer))
