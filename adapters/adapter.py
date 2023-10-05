import sys
import os

from collections import OrderedDict

from torch import Tensor
from torch.nn import GELU, init, Linear, Module, Sequential
sys.path.append(os.path.dirname(__file__))

from mixin import AdapterMixin

__all__ = ["Adapter"]


class Adapter(AdapterMixin, Module):
    """

    Parameter-Efficient Transfer Learning for NLP
    by Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone,
    Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly

    """

    def __init__(self, adapter_size: int):
        super().__init__()
        self.adapter = Sequential(
            OrderedDict(
                [
                    ("A", Linear(2048, adapter_size, bias=True)),
                    ("act", GELU()),
                    ("B", Linear(adapter_size, 2048, bias=True)),
                ]
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        # TODO check if gaussian init is the standard
        for param in self.adapter.parameters():
            init.zeros_(param)

    def forward(self, input: Tensor) -> Tensor:
        
        return self.adapter(input) + input