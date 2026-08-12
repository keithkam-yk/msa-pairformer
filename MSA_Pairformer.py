"""Deprecated import alias for the ``msa_pairformer`` package.

Releases up to 1.0.2 shipped the package as ``MSA_Pairformer``. It was renamed to
``msa_pairformer`` to follow PEP 8, which breaks code such as::

    from MSA_Pairformer.model import MSAPairformer

This module keeps that spelling working. It cannot be a package directory of its own:
on case-insensitive filesystems (macOS, Windows) a directory named ``MSA_Pairformer``
is the same directory as ``msa_pairformer``. So instead it is a single top-level module
that installs an alias finder mapping every ``MSA_Pairformer.*`` name onto the
identically-named ``msa_pairformer.*`` module. The alias resolves to the *same* module
object, not a second copy, so classes imported through either name are identical and
``isinstance`` keeps working across the two spellings.

This shim is deprecated and will be removed in a future release; import
``msa_pairformer`` directly.
"""

import sys
import warnings
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

_OLD_NAME = "MSA_Pairformer"
_NEW_NAME = "msa_pairformer"


class _AliasLoader(Loader):
    """Loader that hands back the already-imported module under its real name."""

    def __init__(self, real_name: str):
        self._real_name = real_name
        self._real_spec = None

    def create_module(self, spec):
        module = import_module(self._real_name)
        # importlib overwrites __spec__ unconditionally in _init_module_attrs, even with
        # override=False; stash the real one so exec_module can put it back.
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module):
        # The module has already been executed under its real name. Restore the __spec__
        # importlib just stamped with the aliased name, so that the real module keeps
        # reporting its real identity (pickling of nn.Modules depends on this).
        if self._real_spec is not None:
            module.__spec__ = self._real_spec


class _AliasFinder(MetaPathFinder):
    """Resolve ``MSA_Pairformer.<sub>`` to the real ``msa_pairformer.<sub>`` module."""

    prefix = _OLD_NAME + "."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix):
            return None
        real_name = _NEW_NAME + "." + fullname[len(self.prefix):]
        try:
            module = import_module(real_name)
        except ModuleNotFoundError as exc:
            if exc.name == real_name:
                # No such submodule: let the normal machinery raise the error for the
                # name the user actually asked for
                return None
            # The submodule exists but one of its imports is missing (e.g. an optional
            # extra such as fair-esm). Report that, not a misleading "no such module".
            raise
        search_locations = getattr(module, "__path__", None)
        return ModuleSpec(
            fullname,
            _AliasLoader(real_name),
            origin=getattr(module, "__file__", None),
            is_package=search_locations is not None,
        )


def _install_finder():
    for finder in sys.meta_path:
        if isinstance(finder, _AliasFinder):
            return
    # Must go first: with __path__ pointing anywhere real, the default PathFinder would
    # happily load a second, independent copy of every submodule under the old name.
    sys.meta_path.insert(0, _AliasFinder())


_install_finder()

# Marks this module as a package so that ``import MSA_Pairformer.model`` is attempted at
# all. It is deliberately empty: every submodule is resolved by _AliasFinder, and an
# empty path means a stray lookup fails loudly instead of silently loading a duplicate.
__path__: list[str] = []

_package = import_module(_NEW_NAME)
__version__ = getattr(_package, "__version__", None)

warnings.warn(
    f"The '{_OLD_NAME}' package was renamed to '{_NEW_NAME}' after version 1.0.2. "
    f"'{_OLD_NAME}' still works through a compatibility shim but is deprecated and will "
    f"be removed in a future release; use 'import {_NEW_NAME}' instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    """Forward attribute access to the real package (``MSA_Pairformer.<attr>``)."""
    return getattr(_package, name)
