from __future__ import annotations

import torch
import torch.nn as nn
import os
import json

class LSTM(nn.Module):
    ''' A sequence-to-sequence model using an LSTM architecture. '''

    def __init__(self, n_src_vocab: int, n_trg_vocab: int, src_pad_idx: int, trg_pad_idx: int,
    embedding_dim: int = 512, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = False, trg_emb_prj_weight_sharing: bool = True, positional_encoding: bool = True):

        super(LSTM, self).__init__()

        # Validation
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive integers.")
        if num_layers <= 0:
            raise ValueError("Number of layers must be a positive integer.")
        if not (0 <= dropout < 1):
            raise ValueError("Dropout rate must be between 0 and 1.")
        if trg_emb_prj_weight_sharing and embedding_dim != hidden_dim:
            raise ValueError("When using the weight sharing mechanism, the hidden dimension must be equal to the embedding dimension.")

        self._params_dict = {
            "n_src_vocab": n_src_vocab,
            "n_trg_vocab": n_trg_vocab,
            "src_pad_idx": src_pad_idx,
            "trg_pad_idx": trg_pad_idx,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "trg_emb_prj_weight_sharing": trg_emb_prj_weight_sharing,
            "positional_encoding": positional_encoding
        }

        # Save padding indices
        self._src_pad_idx = src_pad_idx
        self._trg_pad_idx = trg_pad_idx

        # Embeddings
        self._embedding_src = nn.Embedding(n_src_vocab, embedding_dim, padding_idx=src_pad_idx)
        self._embedding_trg = nn.Embedding(n_trg_vocab, embedding_dim, padding_idx=trg_pad_idx)
        self._embedding_dropout = nn.Dropout(dropout)

        # Normalization
        self.layer_norm_enc = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.layer_norm_dec = nn.LayerNorm(hidden_dim)

        # Encoder
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embedding_dim) if positional_encoding else None

        # Decoder
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False, # Decoder is always uni-directional
            batch_first=True
        )

        # Projection Layer
        self.fc = nn.Linear(hidden_dim, n_trg_vocab)

        # Initialize Weights
        self.initialize_weights()

        # Weight Sharing
        if trg_emb_prj_weight_sharing:
            self.fc.weight = self._embedding_trg.weight

    def initialize_weights(self):
        """ Initialize the model weights. """
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:  # Check if tensor has 2 or more dimensions
                nn.init.xavier_normal_(param)
            elif "bias" in name:  # Bias terms are typically 1D
                nn.init.constant_(param, 0)
   
    def forward(self, src_seq, trg_seq):
        # Source Embedding
        src_emb = self._embedding_src(src_seq)
        src_emb = self._embedding_dropout(src_emb)
        src_emb = self.positional_encoding(src_emb) if self.positional_encoding else src_emb

        # Encode Source Sequence
        enc_output, (hidden, cell) = self.encoder(src_emb)
        enc_output = self.layer_norm_enc(enc_output)

        if self.encoder.bidirectional:
            # Combine bidirectional hidden states
            hidden = combine_bidirectional_states(hidden, self.encoder.num_layers, self._params_dict["hidden_dim"])
            cell = combine_bidirectional_states(cell, self.encoder.num_layers, self._params_dict["hidden_dim"])

        # Target Embedding
        trg_emb = self._embedding_trg(trg_seq)
        trg_emb = self._embedding_dropout(trg_emb)
        trg_emb = self.positional_encoding(trg_emb) if self.positional_encoding else trg_emb

        # Decode Target Sequence
        dec_output, _ = self.decoder(trg_emb, (hidden, cell))
        dec_output = self.layer_norm_dec(dec_output)

        flattened_dec_output = dec_output.reshape(-1, dec_output.size(-1))

        # Project to Vocabulary Space
        seq_logit = self.fc(flattened_dec_output)

        return seq_logit.view(-1, seq_logit.size(1))

    def save_weights_as(self, dir: str, filename: str):
        """ Save the model weights to a file. """
        torch.save(self.state_dict(), os.path.join(dir, f"{filename}.pth"))

    def load_weights_from(self, path: str):
        """ Load the model weights from a file. """
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_params_to(self, path: str):
        """ Save the model parameters to a JSON file. """
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(self._params_dict, f, sort_keys=True, indent=4)

    @staticmethod
    def from_params(path: str) -> LSTM:
        """ Load the model parameters from a JSON file and initialize the model. """
        with open(path, "r") as f:
            params = json.load(f)

        return LSTM(**params)

# Utility functions
@staticmethod
def get_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """ Create a mask for padding elements in a sequence. """
    return (seq != pad_idx).unsqueeze(-2)

def combine_bidirectional_states(states: torch.Tensor, num_layers, hidden_dim) -> torch.Tensor:
    """ Combine bidirectional hidden or cell states. """
    states = states.view(num_layers, 2, -1, hidden_dim)
    states = states[:, 0, :, :] + states[:, 1, :, :]  # Combine forward and backward
    return states.view(num_layers, -1, hidden_dim)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

        # Compute the positional encodings once
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input embeddings.
        :param x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        :return: Tensor of the same shape with positional encodings added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
