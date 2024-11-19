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
            print(f"Added {num_added_tokens} tokens to the tokenizer and resized embeddings.")

    def _load_tokens(self, filename: str) -> List:
        with open(os.path.join(self._dataset_path, filename), "r") as f:
            return json.load(f)

    @property
    def padding_idx_context(self) -> int:
        """Returns the padding token index for context."""
        return self._padding_idx
    
    @property
    def padding_idx_target(self) -> int:
        """Returns the padding token index for targets."""
        return self._padding_idx
    
    @property
    def start_idx_target(self) -> int:
        return self._start_idx

    @property
    def stop_idx_target(self) -> int:
        """Returns the stop token index."""
        return self._stop_idx

    @property
    def size_context_vocab(self) -> int:
        """Returns the size of the vocabulary for context."""
        return self.tokenizer.vocab_size

    
    def stoi_targets(self, input_text: str) -> List[int]:
        """
        Converts a target text string into a list of token IDs.
        :param input_text: Text to encode.
        :return: List of token IDs.
        """
        tokens = self.tokenizer.tokenize(input_text)
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def stoi_context(self, input_text: str) -> List[int]:
        """
        Converts a context text string into a list of token IDs.
        :param input_text: Text to encode.
        :return: List of token IDs.
        """
        tokens = self.tokenizer.tokenize(input_text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def itos_targets(self, token_ids: List[int]) -> str:
        """
        Converts a list of token IDs back into a target text string.
        :param token_ids: List of token IDs.
        :return: Decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def itos_context(self, token_ids: List[int]) -> str:
        """
        Converts a list of token IDs back into a context text string.
        :param token_ids: List of token IDs.
        :return: Decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)