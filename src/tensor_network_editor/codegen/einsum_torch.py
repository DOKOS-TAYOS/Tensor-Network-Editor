from __future__ import annotations

from ..models import EngineName
from .einsum import BaseEinsumCodeGenerator


class EinsumTorchCodeGenerator(BaseEinsumCodeGenerator):
    engine = EngineName.EINSUM_TORCH
    import_line = "import torch"
    module_alias = "torch"
    zero_initializer_suffix = ", dtype=torch.float32"
