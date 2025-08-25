import torch
import torch.nn as nn

class ExpertMLP(nn.Module):
    """
    Tiny MLP expert: d -> 4d -> d with GELU.
    """
    def __init__(self, d_model: int, d_hidden: int = None):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        self.up = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.down = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))
