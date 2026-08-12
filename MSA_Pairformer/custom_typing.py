"""Shape-aware tensor annotations backed by jaxtyping.

Annotations are written as ``Float['b s n d']`` rather than jaxtyping's native
``Float[Tensor, 'b s n d']``; the ``TorchTyping`` shim below supplies the Tensor
argument so that call sites stay readable.

Runtime shape checking is **off by default** because it adds per-call overhead
on the model's hot path. Enable it by setting ``MSA_PAIRFORMER_TYPECHECK=1``
(or true/yes/on) before importing the package, and install the optional
dependency::

    pip install 'msa-pairformer[typecheck]'
    MSA_PAIRFORMER_TYPECHECK=1 python your_script.py

When enabled, ``@typecheck`` becomes ``jaxtyped(typechecker=beartype)``, which
validates dtypes and shapes and, critically, checks that repeated axis names are
*consistent* within a single decorated call: passing an MSA of ``[b s n d]`` and
a pair representation of ``[b m m dp]`` raises rather than silently broadcasting.

Note that static analysers cannot model these annotations at all -- ``Float['b s
n d']`` is a subscript expression, not a type expression -- so ruff's F722 and
ty's invalid-type-form are suppressed in pyproject.toml.
"""

from environs import Env
from jaxtyping import Bool as JaxBool
from jaxtyping import Float as JaxFloat
from jaxtyping import Int as JaxInt
from jaxtyping import Shaped as JaxShaped
from jaxtyping import jaxtyped
from torch import Tensor

_env = Env()
_env.read_env()


def always(value):
    def inner(*args, **kwargs):
        return value
    return inner

def identity(t):
    return t

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str) -> type[Tensor]:
        return self.abstract_dtype[Tensor, shapes]

# PyTorch-compatible type annotations
Shaped = TorchTyping(JaxShaped)
Float = TorchTyping(JaxFloat)
Int = TorchTyping(JaxInt)
Bool = TorchTyping(JaxBool)

# Runtime shape checking is opt-in; see the module docstring.
should_typecheck = _env.bool("MSA_PAIRFORMER_TYPECHECK", False)

if should_typecheck:
    try:
        from beartype import beartype
        from beartype.door import is_bearable
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "MSA_PAIRFORMER_TYPECHECK is set but beartype is not installed. "
            "Install it with: pip install 'msa-pairformer[typecheck]'"
        ) from exc

    typecheck = jaxtyped(typechecker=beartype)
    beartype_isinstance = is_bearable
else:
    typecheck = identity
    beartype_isinstance = always(True)

__all__ = [
    'Bool',
    'Float',
    'Int',
    'Shaped',
    'beartype_isinstance',
    'should_typecheck',
    'typecheck',
]
