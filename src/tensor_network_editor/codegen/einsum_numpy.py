"""NumPy einsum code generator."""

from __future__ import annotations

from ..models import EngineName
from .einsum import BaseEinsumCodeGenerator


class EinsumNumpyCodeGenerator(BaseEinsumCodeGenerator):
    """Generate NumPy-based einsum code."""

    engine = EngineName.EINSUM_NUMPY
    import_line = "import numpy as np"
    module_alias = "np"
    zero_initializer_suffix = ", dtype=float"
