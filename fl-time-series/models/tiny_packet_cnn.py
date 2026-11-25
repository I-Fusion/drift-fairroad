"""
Lightweight 1D-CNN classifier for MAVLink packet windows.

Designed for onboard / edge drones: fewer parameters than the 2-layer LSTM (~54k),
fixed receptive field over short packet sequences, no recurrent state.
"""
import torch
import torch.nn as nn


class TinyPacketCNN(nn.Module):
    """
    Input: (batch, seq_len, input_size) packet features per timestep.
    Uses depthwise-separable style stacked Conv1d + global average pool.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        seq_len: int = 50,
        channels: tuple = (24, 48, 64),
        dropout: float = 0.15,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.dropout = dropout
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            nn.Conv1d(input_size, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c3, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T)
        if x.dim() == 3 and x.size(-1) == self.input_size:
            x = x.transpose(1, 2)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.head(x)

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
