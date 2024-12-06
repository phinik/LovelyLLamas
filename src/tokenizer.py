import os
from typing import List
from transformers import BertTokenizer
import json

class BertBasedTokenizer:
    def __init__(self, dataset_path: str, bert_model_name: str = "bert-base-uncased"):
        self._dataset_path = dataset_path

        # Load BERT tokenizer
        self._bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Load context tokens from JSON (if needed)
        self._context_tokens = self._load_tokens(os.path.join(self._dataset_path, "context_tokens.json"))
        self._context_tokens += ["<padding>"]

        self._target_tokens = self._load_tokens(os.path.join(self._dataset_path, "target_tokens.json"))
        self._target_tokens += ["<start>", "<stop>", "<punkt>", "<komma>", "<space>", "<padding>"]

        # Build context vocabulary mappings
        self._stoi_context = {token: i for i, token in enumerate(self._context_tokens)}
        self._stoi_targets = {token: i for i, token in enumerate(self._target_tokens)}

        self._itos_context = {i: token for token, i in self._stoi_context.items()}
        self._itos_targets = {i: token for token, i in self._stoi_targets.items()}

    def _load_tokens(self, filename: str) -> List[str]:
        with open(filename, "r", encoding='utf-8') as f:
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

    def stoi_context(self, input: str) -> List[int]:
        """Convert context input to indices using context vocabulary."""
        words = input.split(",")
        for word in words:
            if word not in self._stoi_context:
                self._add_new_context_token(word)
        return [self._stoi_context[word] for word in words]

    def itos_context(self, input: List[int]) -> str:
        """Convert context indices back to string."""
        return ", ".join([self._itos_context[idx] for idx in input])

    def stoi_targets(self, input: str) -> List[int]:
        """
        Convert target input to indices using BERT tokenizer.
        This uses BERT's subword embeddings, so it handles tokenization internally.
        """
        return self._bert_tokenizer.encode(input, add_special_tokens=True)

    def itos_targets(self, input: List[int]) -> str:
        """
        Convert target indices back to string using BERT tokenizer.
        """
        return self._bert_tokenizer.decode(input, skip_special_tokens=True)

    def _add_new_context_token(self, token: str):
        """
        Add a new token to the context vocabulary and update the JSON file.
        """
        context_file = os.path.join(self._dataset_path, "context_tokens.json")
        with open(context_file, "r") as f:
            context_tokens = json.load(f)

        context_tokens.append(token)
        with open(context_file, "w") as f:
            json.dump(context_tokens, f)

        # Update context vocab mappings
        self._context_tokens = context_tokens + ["<padding>"]
        self._stoi_context = {t: i for i, t in enumerate(self._context_tokens)}
        self._itos_context = {i: t for t, i in self._stoi_context.items()}
