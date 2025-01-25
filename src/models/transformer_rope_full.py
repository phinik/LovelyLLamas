from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torchtune.modules import RotaryPositionalEmbeddings
from typing import Dict, Optional, Union, Callable


class FullRoPETransformer(nn.Module):
    NAME = "FullRoPETransformer"

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
            "batch_first": True,
            "name": self.NAME
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
        
        self._pos_enc = RoPE(
            d_model=self._params["d_model"], 
            n_head=self._params["n_head"], 
            n_position=self._params["n_position"]
        )

        self._src_emb_dropout = RoPEDropout(self._params["dropout"])
        self._tgt_emb_dropout = RoPEDropout(self._params["dropout"])

        self._model = nn.Transformer(
            d_model=self._params["d_model"],
            nhead=self._params["n_head"],
            custom_encoder=EncoderFactory.get(self._params),
            custom_decoder=DecoderFactory.get(self._params),
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

        # Weight sharing between embedding and final projection
        self._final_projection.weight = self._tgt_word_emb.weight

    def forward(self, src_seq, tgt_seq):
        src_pad_mask = self._get_pad_mask(src_seq, self._params["src_pad_idx"])
        trg_pad_mask = self._get_pad_mask(tgt_seq, self._params["tgt_pad_idx"])
        trg_att_mask = self._get_tgt_mask(tgt_seq)
        
        src_emb = self._src_word_emb(src_seq)
        src_emb_rope = self._pos_enc(src_emb)
        src_emb, src_emb_rope = self._src_emb_dropout(src_emb, src_emb_rope)
        src_emb_stack = torch.stack([src_emb_rope, src_emb_rope])
                
        tgt_emb = self._tgt_word_emb(tgt_seq)
        tgt_emb_rope = self._pos_enc(tgt_emb)
        tgt_emb, tgt_emb_rope = self._tgt_emb_dropout(tgt_emb, tgt_emb_rope)
        tgt_emb_stack = torch.stack([tgt_emb_rope, tgt_emb_rope])

        x = self._model(
            src_emb_stack, 
            tgt_emb_stack, 
            tgt_mask=trg_att_mask, 
            src_key_padding_mask=src_pad_mask, 
            tgt_key_padding_mask=trg_pad_mask
        )

        x = self._final_layer_norm(x)
        x = self._final_projection(x)

        return x.view(-1, self._params["tgt_vocab_size"])

    @property
    def name(self):
        return self.NAME
    
    def save_weights_as(self, dir: str, filename: str):
        torch.save(self.state_dict(), os.path.join(dir, f"{filename}.pth"))

    def load_weights_from(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_params_to(self, dir: str):
        with open(os.path.join(dir, "params.json"), "w") as f:
            json.dump(self._params, f, sort_keys=True, indent=4)

    @staticmethod
    def from_params(path: str) -> FullRoPETransformer:
        with open(path, 'r') as f:
            params = json.load(f)

        return FullRoPETransformer(**params)

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
        encoder_norm = nn.LayerNorm(params["d_model"])
            
        return RoPETransformerEncoder(
            params=params,
            num_layers=params["n_layers"], 
            norm=encoder_norm, 
        )


class DecoderFactory:
    @staticmethod
    def get(params: Dict):        
        decoder_norm = nn.LayerNorm(params["d_model"])
            
        return RoPETransformerDecoder(
            params=params,
            num_layers=params["n_layers"], 
            norm=decoder_norm
        )


# The following classes were taken from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
# (last accessed 23/12/2024) and modified to incorporate RoPE
class RoPETransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        params: Dict,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        mask_check: bool = True,
    ) -> None:
        super().__init__(
            encoder_layer=nn.TransformerEncoderLayer(d_model=params["d_model"], nhead=params["n_head"]),  # dummy, will be replaced below
            num_layers=num_layers,  # dummy, will be replaced below
            norm=norm,
            enable_nested_tensor=not params["norm_first"],
            mask_check=mask_check
        )
        
        layers = [RoPETransformerEncoderLayer(
            d_model=params["d_model"],
            nhead=params["n_head"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            activation=params["activation"],
            batch_first=params["batch_first"],
            norm_first=params["norm_first"]
        )]

        for i in range(num_layers - 1):
            layers.append(nn.TransformerEncoderLayer(
                d_model=params["d_model"],
                nhead=params["n_head"],
                dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"],
                activation=params["activation"],
                batch_first=params["batch_first"],
                norm_first=params["norm_first"]
            ))
        
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers
        

class RoPETransformerDecoder(nn.TransformerDecoder):
    def __init__(
        self,
        params: Dict,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            decoder_layer=nn.TransformerDecoderLayer(d_model=params["d_model"], nhead=params["n_head"]),  # dummy, will be replaced below,
            num_layers=num_layers,  # dummy, will be replaced below
            norm=norm
        )
        
        layers = [RoPETransformerDecoderLayer(
            d_model=params["d_model"],
            nhead=params["n_head"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            activation=params["activation"],
            batch_first=params["batch_first"],
            norm_first=params["norm_first"]
        )]

        for i in range(num_layers - 1):
            layers.append(nn.TransformerDecoderLayer(
                d_model=params["d_model"],
                nhead=params["n_head"],
                dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"],
                activation=params["activation"],
                batch_first=params["batch_first"],
                norm_first=params["norm_first"]
            ))
        
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers
        

class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self._normRoPE = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
                
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            orig_x = x[0, ...].squeeze()
            rope_x = x[1, ...].squeeze()
            
            normed_orig_x = self.norm1(orig_x)
            normed_rope_x = self._normRoPE(rope_x)
            
            x = orig_x + self._sa_block(
                normed_orig_x, normed_rope_x, src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            raise RuntimeError("Wrong block")

        return x


    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        rope_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            rope_x,  # RoPE
            rope_x,  # RoPE
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)


class RoPETransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype
        )
        
        self._normRoPE = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            orig_x = x[0, ...].squeeze()
            rope_x = x[1, ...].squeeze()
            
            normed_orig_x = self.norm1(orig_x)
            normed_rope_x = self._normRoPE(rope_x)
            
            x = orig_x + self._sa_block(
                normed_orig_x, normed_rope_x, tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            raise RuntimeError("Wrong block")

        return x


    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        rope_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            rope_x,  # RoPE
            rope_x,  # RoPE
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)


class RoPE(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_position: int = 200):
        super().__init__()

        assert d_model % n_head == 0, "d_model must be a multiple of nhead"

        self._n_head = n_head
        self._head_dim = d_model // n_head
        
        self._rope = RotaryPositionalEmbeddings(self._head_dim, n_position)
        
    def forward(self, x):
        return self._rope(x.view(x.shape[0], x.shape[1], self._n_head, self._head_dim)).view(x.shape)
    

class RoPEDropout(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()

        self._prob = 1.0 - dropout

    def forward(self, x, rope_x):
        if self.training:
            p_map = torch.ones_like(x) * self._prob
            mask = torch.bernoulli(p_map)

            return x * mask, rope_x * mask
        else:
            return x, rope_x
