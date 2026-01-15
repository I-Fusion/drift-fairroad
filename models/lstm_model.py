"""
LSTM Model for Time Series Prediction

Simple LSTM model for GPS+IMU sensor fusion.
Users can create their own model files following this template.
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based model for time series prediction.

    To create your own model:
    1. Create a new file like 'my_model.py' in the models/ folder
    2. Define a class with __init__ and forward methods
    3. Update MODEL_PATH in config.py to point to your model file
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = None,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features (default: same as input)
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size or input_size
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        last_hidden = h_n[-1]

        # Fully connected layers
        out = self.fc(last_hidden)

        return out

    def get_weights(self):
        """Get model weights as state dict"""
        return self.state_dict()

    def set_weights(self, weights):
        """Set model weights from state dict"""
        self.load_state_dict(weights)

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
