import json
import os
from transformers import BertTokenizer
from typing import List


class Tokenizer:
    def __init__(self, dataset_path: str, model_name: str = "bert-base-german-cased"):
        """
        Initializes the tokenizer using a pre-trained BERT model for German.
        :param model_name: Name of the pre-trained BERT model.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self._dataset_path = dataset_path

        self._target_tokens = list(self.tokenizer.get_vocab().keys())
        self._context_tokens = list(self.tokenizer.get_vocab().keys())

        custom_tokens = ["<start>", "<stop>", "<padding>", "degC", "l_per_sqm", "kmh", "percent"]
        self._target_tokens += custom_tokens

        self._stoi_targets = {token: i for i, token in enumerate(self._target_tokens)}
        self._stoi_context = {token: i for i, token in enumerate(self._context_tokens)}

        # Rebuild the dictionaries for custom tokens
        for idx, token in enumerate(custom_tokens, len(self.tokenizer.get_vocab())):
            self._stoi_targets[token] = idx
            self._stoi_context[token] = idx

        # Invert the dictionaries
        self._itos_targets = {i: token for token, i in self._stoi_targets.items()}
        self._itos_context = {i: token for token, i in self._stoi_targets.items()}

        # Special token IDs
        self._padding_idx = self.tokenizer.pad_token_id
        self._start_idx = self.tokenizer.cls_token_id
        self._stop_idx = self.tokenizer.sep_token_id

    def add_custom_tokens(self, tokens: List[str]):
        """
        Adds custom tokens to the tokenizer.
        :param tokens: List of custom tokens.
        """
        num_added_tokens = self.tokenizer.add_tokens(tokens)
        if num_added_tokens > 0:
            size_vocab = self.tokenizer.vocab_size
            print(f"Added {num_added_tokens} tokens to the tokenizer and resized embeddings.")

    def _load_tokens(self, filename: str) -> List:
        with open(os.path.join(self._dataset_path, filename), "r") as f:
            return json.load(f)

    @property
    def padding_idx_context(self) -> int:
        return self._stoi_context["<padding>"]
    
    @property
    def padding_idx_target(self) -> int:
        return self._stoi_targets["<padding>"]
        
    @property
    def start_idx_target(self) -> int:
        return self._stoi_targets["<start>"]
    
    @property
    def stop_idx_target(self) -> int:
        return self._stoi_targets["<stop>"]
    
    @property
    def size_context_vocab(self) -> int:
        return max(self._stoi_context.values()) + 1
    
    @property
    def size_target_vocab(self) -> int:
        return max(self._stoi_targets.values()) + 1

    def stoi_targets(self, token: str) -> int:
        """
        Maps a target token to its corresponding index.
        :param token: Token to map.
        :return: Corresponding index or raises KeyError if token is not found.
        """
        return self._stoi_targets.get(token, None)

    def stoi_context(self, token: str) -> int:
        """
        Maps a context token to its corresponding index.
        :param token: Token to map.
        :return: Corresponding index or raises KeyError if token is not found.
        """
        return self._stoi_context.get(token, None)

    def itos_targets(self, idx: int) -> str:
        """
        Maps a target index to its corresponding token.
        :param idx: Index to map.
        :return: Corresponding token or raises KeyError if index is not found.
        """
        return self._itos_targets.get(idx, None)

    def itos_context(self, idx: int) -> str:
        """
        Maps a context index to its corresponding token.
        :param idx: Index to map.
        :return: Corresponding token or raises KeyError if index is not found.
        """
        return self._itos_context.get(idx, None)