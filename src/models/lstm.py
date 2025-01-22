from __future__ import annotations

import torch
import torch.nn as nn
import os
import json
import math

class LSTM(nn.Module):
    ''' A sequence-to-sequence model using an LSTM architecture. '''

    def __init__(self,
                 name: str,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 src_pad_idx: int,
                 tgt_pad_idx: int,
                 embedding_dim: int = 512,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 tgt_emb_prj_weight_sharing: bool = True,
                 positional_encoding: bool = True,
                 attention: bool = False,
                 label_smoothing: float = 0.0,
                 use_layer_norm: bool = True):

        super(LSTM, self).__init__()

        # Validation
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive integers.")
        if num_layers <= 0:
            raise ValueError("Number of layers must be a positive integer.")
        if not (0 <= dropout < 1):
            raise ValueError("Dropout rate must be between 0 and 1.")
        if tgt_emb_prj_weight_sharing and embedding_dim != hidden_dim:
            raise ValueError("When using the weight sharing mechanism, the hidden dimension must be equal to the embedding dimension.")
        if not (0 <= label_smoothing < 1):
            raise ValueError("Label smoothing must be between 0 and 1.")

        self._params_dict = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "src_pad_idx": src_pad_idx,
            "tgt_pad_idx": tgt_pad_idx,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "tgt_emb_prj_weight_sharing": tgt_emb_prj_weight_sharing,
            "positional_encoding": positional_encoding,
            "attention": attention,
            "label_smoothing": label_smoothing,
            "use_layer_norm": use_layer_norm,
            "name": name
        }

        self.name = name

        # Save padding indices
        self._src_pad_idx = src_pad_idx
        self._tgt_pad_idx = tgt_pad_idx

        # Embeddings
        self._embedding_src = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=src_pad_idx)
        self._embedding_tgt = nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=tgt_pad_idx)
        self._embedding_dropout = nn.Dropout(dropout)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embedding_dim) if positional_encoding else None

        # Layer Normalization (optional)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
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

        # Decoder
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False, # Decoder is always uni-directional
            batch_first=True
        )

        # Attention (optional)
        self.attention = BahdanauAttention(hidden_dim) if attention else None

        # Projection Layer
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

        # Initialize Weights
        self.initialize_weights()

        # Weight Sharing
        if tgt_emb_prj_weight_sharing:
            self.fc.weight = self._embedding_tgt.weight

    def initialize_weights(self):
        """ Initialize the model weights. """
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:  # Check if tensor has 2 or more dimensions
                nn.init.xavier_normal_(param)
            elif "bias" in name:  # Bias terms are typically 1D
                nn.init.constant_(param, 0)

    def forward(self, src_seq, tgt_seq):
        # Source Embedding
        src_emb = self._embedding_src(src_seq)
        src_emb = self._embedding_dropout(src_emb)
        src_emb = self.positional_encoding(src_emb) if self.positional_encoding else src_emb

        # Encode Source Sequence
        enc_output, (hidden, cell) = self.encoder(src_emb)
        if self.use_layer_norm:
            enc_output = self.layer_norm_enc(enc_output)

        if self.encoder.bidirectional:
            # Combine bidirectional hidden states
            hidden = combine_bidirectional_states(hidden, self.encoder.num_layers, self._params_dict["hidden_dim"])
            cell = combine_bidirectional_states(cell, self.encoder.num_layers, self._params_dict["hidden_dim"])

        # Target Embedding
        tgt_emb = self._embedding_tgt(tgt_seq)
        tgt_emb = self._embedding_dropout(tgt_emb)
        tgt_emb = self.positional_encoding(tgt_emb) if self.positional_encoding else tgt_emb

        # Decode Target Sequence
        dec_output, _ = self.decoder(tgt_emb, (hidden, cell))
        if self.use_layer_norm:
            dec_output = self.layer_norm_dec(dec_output)

        # Apply Attention if enabled
        if self.attention:
            dec_output, _ = self.attention(dec_output, enc_output)

        # Project to Vocabulary Space
        seq_logit = self.fc(dec_output) # (B, T, vocab_size)

        # Reshape to match CrossEntropyLoss expectations
        seq_logit = seq_logit.view(-1, seq_logit.size(-1)) # (B * T, vocab_size)

        return seq_logit

    def save_weights_as(self, dir: str, filename: str):
        """ Save the model weights to a file. """
        torch.save(self.state_dict(), os.path.join(dir, f"{filename}.pth"))

    def load_weights_from(self, path: str):
        """ Load the model weights from a file. """
        self.load_state_dict(torch.load(path))

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
def get_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """ Create a mask for padding elements in a sequence. """
    return (seq != pad_idx).unsqueeze(-2)

def combine_bidirectional_states(states: torch.Tensor, num_layers, hidden_dim) -> torch.Tensor:
    """ Combine bidirectional hidden or cell states. """
    states = states.view(num_layers, 2, -1, hidden_dim)
    states = states[:, 0, :, :] + states[:, 1, :, :]  # Combine forward and backward
    return states.view(num_layers, -1, hidden_dim)

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

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)

    def forward(self, query: torch.Tensor, values: torch.Tensor):
        """
        Compute the attention weights and context vector.
        :param query: Decoder hidden state (batch_size, seq_len, hidden_dim)
        :param values: Encoder outputs (batch_size, seq_len, hidden_dim)
        :return: Context vector and attention weights
        """
        query = self.query_layer(query)
        keys = self.key_layer(values)
        energy = torch.tanh(query.unsqueeze(2) + keys.unsqueeze(1))
        attention = torch.softmax(self.energy_layer(energy).squeeze(-1), dim=-1)
        context = torch.bmm(attention, values)
        return context, attention
