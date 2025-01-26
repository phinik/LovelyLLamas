import json

from typing import Dict

from .transformer import Transformer
from .transformer_rope import RoPETransformer
from .transformer_rope_dropout import RoPETransformerDropout
from .transformer_rope_ddropout import RoPETransformerDDropout
from .transformer_rope_full import FullRoPETransformer


class TransformerFactory:
    def __init__(self):
        ...

    @staticmethod
    def from_dict(model_type: str, params: Dict):
        if model_type == "og_transformer":
            return Transformer(**params)
        elif model_type == "rope_transformer":
            return RoPETransformer(**params)
        elif model_type == "rope_transformer_dropout":
            return RoPETransformerDropout(**params)
        elif model_type == "rope_transformer_ddropout":
            return RoPETransformerDDropout(**params)
        elif model_type == "full_rope_transformer":
            return FullRoPETransformer(**params)
        else:
            raise KeyError(f"Unkonwn 'model_type' {model_type}")
        
    @staticmethod
    def from_file(path: str):
        with open(path, "r") as f:
            config = json.load(f)
            model_type = config["name"]

        if model_type == Transformer.NAME:
            return Transformer.from_params(path)
        elif model_type == RoPETransformer.NAME:
            return RoPETransformer.from_params(path)
        elif model_type == RoPETransformerDropout.NAME:
            return RoPETransformerDropout.from_params(path)
        elif model_type == RoPETransformerDDropout.NAME:
            return RoPETransformerDDropout.from_params(path)
        elif model_type == RoPETransformerDDropout.NAME:
            return FullRoPETransformer.from_params(path)
        else:
            raise KeyError(f"Unkonwn 'model_type' {model_type}")
