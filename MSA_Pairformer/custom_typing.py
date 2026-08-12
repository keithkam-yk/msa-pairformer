from jaxtyping import Bool as JaxBool
from jaxtyping import Float as JaxFloat
from jaxtyping import Int as JaxInt
from jaxtyping import Shaped as JaxShaped
from torch import Tensor


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

# PyTorch-compatible type annotations.
#
# These build jaxtyping annotations such as Float[Tensor, 'b s n d'] at runtime.
# Annotations of the form Float['b s n d'] are a subscript expression rather than
# a type expression, so they sit outside the typing spec and no static checker can
# model them: ruff reads the shape string as a malformed forward reference (F722)
# and ty reports an invalid subscript. Both rules are suppressed in pyproject.toml.
# The annotations are inert at runtime as well, since typecheck is identity below.
Shaped = TorchTyping(JaxShaped)
Float = TorchTyping(JaxFloat)
Int = TorchTyping(JaxInt)
Bool = TorchTyping(JaxBool)

# Type checking is disabled by default
should_typecheck = False
typecheck = identity
beartype_isinstance = always(True)
__all__ = [
    'Bool',
    'Float',
    'Int',
    'beartype_isinstance',
    'typecheck'
]
