"""Variant-effect prediction helpers for the Figure 5B ProteinGym script.

The modules here were once loaded by appending this directory to `sys.path`,
which only worked when the interpreter started in `Figure5/` and put five
generic names (`weights`, `msa_utils`, ...) into the top-level module
namespace.  Keeping them in a package leaves one importable name and lets the
figure script run from anywhere.

Nothing is re-exported: `fitness` pulls in torch and `msa_io` pulls in pandas,
and importing the package should not require either.
"""
