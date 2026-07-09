import os
import sys


try:
    from . import _C
except ImportError:
    raise ImportError("Frex C++ extension binaries not compiled. Run `pip install -e .` inside your root directory.")

from .tensor import Tensor
from .optim import SGD
from .loss import cross_entropy_loss
from . import nn

class FrexContext:
    def __init__(self, fast_vram_mb: int = 64, slow_ram_mb: int = 512, device: str = None):
        self.cuda_available = _C.is_cuda_available()
        
        if device is None:
            self.device = "cuda" if self.cuda_available else "cpu"
        else:
            if device == "cuda" and not self.cuda_available:
                print("[Frex Warning] CUDA requested but not found. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = device
                
        # Boot the C++ Backend with the Unified Memory limits
        self._engine = _C.FrexEngine(fast_vram_mb, slow_ram_mb)

    def create_tensor(self, name: str, shape: list, init_val: float = 0.0, requires_grad: bool = False) -> Tensor:
        return Tensor(self, name, shape, init_val, requires_grad)

    def delete(self, name: str):
        self._engine.free_tensor(name)

    def display_hardware_state(self):
        self._engine.print_memory_map()