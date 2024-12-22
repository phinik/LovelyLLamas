from __future__ import annotations

import json
import torch
import torch.nn as nn
import os

from typing import Dict


class Transformer(nn.Module):
    def __init__(
            self, 
            src_vocab_size, 
            tgt_vocab_size, 
            src_pad_idx, 
            tgt_pad_idx,
            d_model=512, 
            dim_feedforward=2048,
            n_layers=6, 
            n_head=8, 
            dropout=0.1, 
            activation="relu",
            n_position=200, 
            **kwargs
        ):
        super().__init__()

        self._params = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "src_pad_idx": src_pad_idx,
            "tgt_pad_idx": tgt_pad_idx,
            "d_model": d_model,
            "dim_feedforward": dim_feedforward,
            "n_layers": n_layers,
            "n_head": n_head,
            "dropout": dropout,
            "activation": activation,
            "n_position": n_position,
            "norm_first": True,
            "batch_first": True
        }

        self._src_word_emb = nn.Embedding(
            num_embeddings=self._params["src_vocab_size"], 
            embedding_dim=self._params["d_model"], 
            padding_idx=self._params["src_pad_idx"]
        )
        
        self._tgt_word_emb = nn.Embedding(
            num_embeddings=self._params["tgt_vocab_size"], 
            embedding_dim=self._params["d_model"], 
            padding_idx=self._params["tgt_pad_idx"]
        )
        
        self._pos_enc = PositionalEncoding(self._params["d_model"])
        self._dropout = nn.Dropout(self._params["dropout"])

        self._model = nn.Transformer(
            d_model=self._params["d_model"],
            nhead=self._params["n_head"],
            custom_encoder=EncoderFactory.get(self._params),
            custom_decoder=DecoderFactory.get(self._params),
            #num_decoder_layers=self._params["n_layers"],
            #num_encoder_layers=self._params["n_layers"], 
            activation=self._params["activation"],
            dim_feedforward=self._params["dim_feedforward"], 
            dropout=self._params["dropout"], 
            norm_first=self._params["norm_first"],
            batch_first=self._params["batch_first"]
           )
                
        self._final_layer_norm = nn.LayerNorm(normalized_shape=self._params["d_model"])
        
        self._final_projection = nn.Linear(
            in_features=self._params["d_model"], 
            out_features=self._params["tgt_vocab_size"]
        )

    def forward(self, src_seq, tgt_seq):
        src_pad_mask = self._get_pad_mask(src_seq, self._params["src_pad_idx"])
        trg_pad_mask = self._get_pad_mask(tgt_seq, self._params["tgt_pad_idx"])
        trg_att_mask = self._get_tgt_mask(tgt_seq)
        
        src_emb = self._src_word_emb(src_seq)
        src_emb = self._pos_enc(src_emb)

        tgt_emb = self._tgt_word_emb(tgt_seq)
        tgt_emb = self._pos_enc(tgt_emb)

        x = self._model(
            src_emb, 
            tgt_emb, 
            tgt_mask=trg_att_mask, 
            src_key_padding_mask=src_pad_mask, 
            tgt_key_padding_mask=trg_pad_mask
        )

        x = self._final_layer_norm(x)
        x = self._final_projection(x)

        return x.view(-1, self._params["tgt_vocab_size"])

    def save_weights_as(self, dir: str, filename: str):
        torch.save(self.state_dict(), os.path.join(dir, f"{filename}.pth"))

    def load_weights_from(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_params_to(self, dir: str):
        with open(os.path.join(dir, "params.json"), "w") as f:
            json.dump(self._params, f, sort_keys=True, indent=4)

    @staticmethod
    def from_params(path: str) -> Transformer:
        with open(path, 'r') as f:
            params = json.load(f)

        return Transformer(**params)

    @staticmethod
    def _get_pad_mask(seq, pad_idx):
        return (seq == pad_idx).to(seq.device)

    @staticmethod
    def _get_tgt_mask(seq):
        _, sequence_length = seq.shape
        
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
        
        return mask.to(seq.device)


class EncoderFactory:
    @staticmethod
    def get(params: Dict):
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=params["d_model"],
                nhead=params["n_head"],
                dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"],
                activation=params["activation"],
                batch_first=params["batch_first"],
                norm_first=params["norm_first"]
            )
        
        encoder_norm = nn.LayerNorm(params["d_model"])
            
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=params["n_layers"], 
            norm=encoder_norm, 
            enable_nested_tensor=not params["norm_first"]
        )


class DecoderFactory:
    @staticmethod
    def get(params: Dict):
        decoder_layer = nn.TransformerDecoderLayer(
                d_model=params["d_model"],
                nhead=params["n_head"],
                dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"],
                activation=params["activation"],
                batch_first=params["batch_first"],
                norm_first=params["norm_first"]
            )
        
        decoder_norm = nn.LayerNorm(params["d_model"])
            
        return nn.TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=params["n_layers"], 
            norm=decoder_norm
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, n_position: int = 200, dropout: float = 0.1):
        super().__init__()
        # Inspired by https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html (22/12/2024)
        
        self._dropout = nn.Dropout(p=dropout)
        
        pos_encoding = torch.zeros(n_position, d_model)  # POS x i
        positions = torch.arange(0., n_position).unsqueeze(1)

        denom = torch.pow(10000, torch.arange(0., d_model, 2) / d_model)

        pos_encoding[:, 0::2] = torch.sin(positions * denom)
        pos_encoding[:, 1::2] = torch.cos(positions * denom)

        pos_encoding = pos_encoding.unsqueeze(0)
        
        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self._dropout(x + self.pos_encoding[:,  :x.shape[1], :])
