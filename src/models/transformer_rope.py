from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torchtune.modules import RotaryPositionalEmbeddings
from typing import Dict, Optional, Union, Callable


class RoPETransformer(nn.Module):
    NAME = "RoPETransformer"

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
        #src_emb = self._pos_enc(src_emb)

        tgt_emb = self._tgt_word_emb(tgt_seq)
        #tgt_emb = self._pos_enc(tgt_emb)

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
    def from_params(path: str) -> RoPETransformer:
        with open(path, 'r') as f:
            params = json.load(f)

        return RoPETransformer(**params)

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
    

# The following classes were taken from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
# (last accessed 23/12/2024) and modified to incorporate RoPE
class RoPETransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        params: Dict,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        mask_check: bool = True,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
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
        self.norm = norm

        self.mask_check = mask_check

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None

        seq_len = nn.modules.transformer._get_seq_len(src, batch_first)
        is_causal = nn.modules.transformer._detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )

        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output


class RoPETransformerDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        params: Dict,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
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
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        output = tgt

        seq_len = nn.modules.transformer._get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = nn.modules.transformer._detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output



class RoPETransformerEncoderLayer(nn.Module):
    __constants__ = ["norm_first"]

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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = nn.modules.transformer._get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        assert d_model % nhead == 0, "d_model must be a multiple of nhead"
        self._rope = RotaryPositionalEmbeddings(d_model // nhead, 200)
        self._n_heads = nhead
        self._head_dim = d_model // nhead

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

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

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ""
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
            src_key_padding_mask is not None or src_mask is not None
        ):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        elif any(
            len(getattr(m, "_forward_hooks", {}))
            + len(getattr(m, "_forward_pre_hooks", {}))
            for m in self.modules()
        ):
            why_not_sparsity_fast_path = "forward pre-/hooks are attached to the module"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                (x.device.type in _supported_device_type) for x in tensor_args
            ):
                why_not_sparsity_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"{_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src
                )
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, self._n_heads, self._head_dim, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        n_heads: int,
        head_dim: int,
        is_causal: bool = False,
    ) -> torch.Tensor:
        roped_x = self._rope(x.view(x.shape[0], x.shape[1], n_heads, head_dim)).view(x.shape)
        x = self.self_attn(
            roped_x,  # RoPE
            roped_x,  # RoPE
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class RoPETransformerDecoderLayer(nn.Module):
    __constants__ = ["norm_first"]

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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = nn.modules.transformer._get_activation_fn(activation)
        else:
            self.activation = activation

        assert d_model % nhead == 0, "d_model must be a multiple of nhead"
        self._rope = RotaryPositionalEmbeddings(d_model // nhead, 200)
        self._n_heads = nhead
        self._head_dim = d_model // nhead

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

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
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, self._n_heads, self._head_dim, tgt_is_causal
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
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        n_heads: int,
        head_dim: int,
        is_causal: bool = False,
    ) -> torch.Tensor:
        roped_x = self._rope(x.view(x.shape[0], x.shape[1], n_heads, head_dim)).view(x.shape)
        x = self.self_attn(
            roped_x,  # RoPE
            roped_x,  # RoPE
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
