from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(LSTM, self).__init__()
        self._params_dict = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def save_weights_as(self, dir: str, filename: str):
        torch.save(self.state_dict(), os.path.join(dir, f"{filename}.pth"))

    def load_weights_from(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_params_to(self, dir: str):
        with open(os.path.join(dir, "params.json"), "w") as f:
            json.dump(self._params_dict, f, sort_keys=True, indent=4)

    @staticmethod
    def from_params(path: str) -> LSTM:
        with open(path, 'r') as f:
            params = json.load(f)

        return LSTM(**params)
