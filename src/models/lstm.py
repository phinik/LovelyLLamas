from __future__ import annotations

import torch
import torch.nn as nn
import os
import json

class LSTM(nn.Module):
    ''' A sequence-to-sequence model using an LSTM architecture. '''

    def __init__(self, n_src_vocab: int, n_trg_vocab: int, src_pad_idx: int, trg_pad_idx: int,
    embedding_dim: int = 512, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = False, trg_emb_prj_weight_sharing: bool = True):

        super(LSTM, self).__init__()

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
            "trg_emb_prj_weight_sharing": trg_emb_prj_weight_sharing
        }

        self._src_pad_idx = src_pad_idx
        self._trg_pad_idx = trg_pad_idx

        self._embedding_src = nn.Embedding(n_src_vocab, embedding_dim, padding_idx=src_pad_idx)
        self._embedding_trg = nn.Embedding(n_trg_vocab, embedding_dim, padding_idx=trg_pad_idx)

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False, # Decoder is always uni-directional
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, n_trg_vocab)
        print(n_src_vocab)

        if trg_emb_prj_weight_sharing:
            if embedding_dim != hidden_dim:
                raise ValueError("When using the weight sharing mechanism, the hidden dimension must be equal to the embedding dimension.")
            self.fc.weight = self._embedding_trg.weight

   
    def forward(self, src_seq, trg_seq):
        # Source Embedding
        src_emb = self._embedding_src(src_seq)

        # Encode Source Sequence
        enc_output, (hidden, cell) = self.encoder(src_emb)

        if self.encoder.bidirectional:
            # Combine bidirectional hidden states
            hidden = hidden.view(self.encoder.num_layers, 2, -1, self._params_dict["hidden_dim"])
            hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]  # Combine forward and backward
            hidden = hidden.view(self.encoder.num_layers, -1, self._params_dict["hidden_dim"])

            # Combine bidirectional cell states
            cell = cell.view(self.encoder.num_layers, 2, -1, self._params_dict["hidden_dim"])
            cell = cell[:, 0, :, :] + cell[:, 1, :, :]  # Combine forward and backward
            cell = cell.view(self.encoder.num_layers, -1, self._params_dict["hidden_dim"])

        # Target Embedding
        trg_emb = self._embedding_trg(trg_seq)

        # Decode Target Sequence
        dec_output, _ = self.decoder(trg_emb, (hidden, cell))

        flattened_dec_output = dec_output.reshape(-1, dec_output.size(-1))

        # Project to Vocabulary Space
        seq_logit = self.fc(flattened_dec_output)

        return seq_logit.view(-1, seq_logit.size(1))

    def save_weights_as(self, dir: str, filename: str):
        torch.save(self.state_dict(), os.path.join(dir, f"{filename}.pth"))

    def load_weights_from(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_params_to(self, path: str):
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(self._params_dict, f, sort_keys=True, indent=4)

    @staticmethod
    def from_params(path: str) -> LSM:
        with open(path, "r") as f:
            params = json.load(f)

        return LSTM(**params)

# Utility functions
def get_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(-2)