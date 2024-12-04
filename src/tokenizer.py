import json
import os
from transformers import BertTokenizer
from transformers import AutoTokenizer
from typing import List


class Tokenizer:
    """Tokenizer for encoding text data using a pre-trained BERT model with custom tokens."""
    def __init__(self, dataset_path: str, model_name: str = "bert-base-german-cased", custom_tokens: List[str] = None):
        if model_name == "bert-base-german-cased":
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._dataset_path = dataset_path

        # Initialize vocabularies
        self._target_tokens = list(self.tokenizer.get_vocab().keys())
        self._context_tokens = list(self.tokenizer.get_vocab().keys())

        # Add custom tokens, avoid duplicates
        self.add_custom_tokens(custom_tokens)

    def add_custom_tokens(self, custom_tokens: List[str] = None):
        # Filter out existing tokens
        new_tokens = [token for token in custom_tokens if token not in self.tokenizer.vocab]
        num_added_tokens = self.tokenizer.add_tokens(new_tokens)

        if num_added_tokens > 0:
            print(f"Added {num_added_tokens} tokens to the tokenizer and resized embeddings.")

        # Update vocabularies with custom tokens
        vocab_size = len(self.tokenizer)
        self._stoi_targets = {token: i for i, token in enumerate(self._target_tokens + custom_tokens)}
        self._stoi_context = {token: i for i, token in enumerate(self._context_tokens + custom_tokens)}
        self._itos_targets = {i: token for token, i in self._stoi_targets.items()}
        self._itos_context = {i: token for token, i in self._stoi_context.items()}

    def encode(self, text: str, truncation: bool = False, max_length: int = 512) -> List[int]:
        """Encodes text into input IDs with padding."""
        encoding = self.tokenizer.encode_plus(text, truncation=truncation, max_length=max_length, return_tensors="pt")

        return encoding["input_ids"].squeeze(0).tolist()
    
    def decode(self, input_ids: List[int]) -> str:
        """Decodes input IDs back into text."""
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
    
    def save_tokenizer(self, path: str):
        """Saves the tokenizer configuration and vocabulary."""
        self.tokenizer.save_pretrained(path)

    def load_tokens(self, filename: str) -> List:
        """Loads tokens from a JSON file."""
        file_path = os.path.join(self._dataset_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Token file not found: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)
        
    # Properties for special token indices
    @property
    def padding_idx(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def start_idx(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<start>")
        
    @property
    def stop_idx(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<stop>")
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)