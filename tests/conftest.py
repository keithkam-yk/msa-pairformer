"""Pin the triangle implementation for the whole suite.

`PairwiseBlock` and `TriangleMultiplication` read the module-level
`CUEQUIVARIANCE_AVAILABLE` at construction, so on a CUDA host they select the
fused kernels automatically. Four cases in `generate_fixtures.py` build those
modules without forcing a path, and every golden in `fixtures/golden.pt` was
recorded on CPU with the vanilla implementation -- so on a GPU host the suite
would build fused modules, feed them CPU tensors, and fail for a reason that
has nothing to do with the code under test.

Forcing vanilla makes the default host-independent: a test that does not say
which kernels it wants gets the same ones everywhere, rather than inheriting
whatever the machine happens to offer.

Tests for which the kernels *are* the subject opt back in explicitly, by
entering `triangle_path` themselves -- `test_correctness.ExecutionPath.replay`
and `test_cuequivariance.py`. `triangle_path` restores the previous value on
exit, so nesting inside this session-wide force is safe.
"""

import pytest

from bench.step import triangle_path


@pytest.fixture(scope="session", autouse=True)
def force_vanilla_triangles():
    """Session-scoped, and that is load-bearing rather than an optimisation.

    pytest instantiates higher-scoped fixtures first, so a function-scoped
    autouse override would run *after* `test_correctness.case_registry`
    (module-scoped) had already resolved the triangle path from the ambient
    constant. On a GPU host that would build the triangle cases with the fused
    kernels while every other case built vanilla -- a split configuration
    reported as one result.
    """
    with triangle_path(False):
        yield
