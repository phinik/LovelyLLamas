import json
import os


from transformers import BertTokenizer
from typing import List
from abc import abstractmethod


from .itokenizer import ITokenizer


class BertTokenizerAdapter(ITokenizer):
    def __init__(self, dataset_path: str):
        """
        Initializes the tokenizer using a pre-trained BERT model for German.
        :param model_name: Name of the pre-trained BERT model.
        """
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
        self._tokenizer.add_tokens(["<degC>", "<city>"], special_tokens=False)
        self._tokenizer.add_tokens(["<start>", "<stop>", "<padding>"], special_tokens=True)

        # Special token IDs
        self._padding_idx = self._tokenizer.convert_tokens_to_ids("<padding>")
        self._start_idx = self._tokenizer.convert_tokens_to_ids("<start>")
        self._stop_idx = self._tokenizer.convert_tokens_to_ids("<stop>")
    
    @property
    def padding_idx(self) -> int:
        """Returns the padding token index"""
        return self._padding_idx
        
    @property
    def start_idx(self) -> int:
        """Returns the start token index."""
        return self._start_idx

    @property
    def stop_idx(self) -> int:
        """Returns the stop token index."""
        return self._stop_idx
    
    @property
    def unknown_idx(self) -> int:
        self._tokenizer.unk_token_id

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        return len(self._tokenizer)

    def add_start_stop_tokens(self, s: str) -> str:
        return f"<start>{s}<stop>"

    def stoi(self, input_text: str) -> List[int]:
        """
        Converts a text string into a list of token IDs.
        :param input_text: Text to encode.
        :return: List of token IDs.
        """
        tokens = self._tokenizer.tokenize(input_text)
        return self._tokenizer.convert_tokens_to_ids(tokens)
    
    def itos(self, token_ids: List[int]) -> str:
        """
        Converts a list of token IDs back into a text string.
        :param token_ids: List of token IDs.
        :return: Decoded text string.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)


class SubsetBertTokenizer(BertTokenizerAdapter):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self._tokens = self._load_tokens(dataset_path)
        self._tokens.append(super().start_idx)
        self._tokens.append(super().stop_idx)
        self._tokens.append(super().padding_idx)
        self._tokens.append(super().unknown_idx)
        self._tokens = set(self._tokens)

        self._bert_to_subset_mapping = {token: idx for idx, token in enumerate(self._tokens)}
        self._subset_to_bert_mapping = {idx: token for token, idx in self._bert_to_subset_mapping.items()}

    def _load_tokens(self, dataset_path: str) -> List[int]:
        with open(os.path.join(dataset_path, self._token_filename), "r") as f:
            return json.load(f)

    @property
    @abstractmethod
    def _token_filename(self) -> str:
        ...

    @property
    def padding_idx(self) -> int:
        return self._bert_to_subset_mapping[super().padding_idx]
        
    @property
    def start_idx(self) -> int:
        return self._bert_to_subset_mapping[super().start_idx]

    @property
    def stop_idx(self) -> int:
        return self._bert_to_subset_mapping[super().stop_idx]
    
    @property
    def unknown_idx(self) -> int:
        self._bert_to_subset_mapping[super().unknown_idx]

    @property
    def vocab_size(self) -> int:
        return len(self._bert_to_subset_mapping)

    def stoi(self, input_text: str) -> List[int]:
        tokens = super().stoi(input_text)
        tokens = [self._bert_to_subset_mapping[token] for token in tokens]
        
        return tokens
    
    def itos(self, token_ids: List[int]) -> str:
        tokens = [self._subset_to_bert_mapping[token] for token in token_ids]
        s = super().itos(tokens)
        
        return s


class SubsetBertTokenizerRepShort(SubsetBertTokenizer):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    @property
    def _token_filename(self) -> str:
        return "rep_short_tokens_bert.json"


class SubsetBertTokenizerGPT(SubsetBertTokenizer):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    @property
    def _token_filename(self) -> str:
        return "gpt_tokens_bert.json"