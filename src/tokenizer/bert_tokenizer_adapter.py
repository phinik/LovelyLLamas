from transformers import BertTokenizer
from typing import List


from .itokenizer import ITokenizer


class BertTokenizerAdapter(ITokenizer):
    def __init__(self):
        """
        Initializes the tokenizer using a pre-trained BERT model for German.
        :param model_name: Name of the pre-trained BERT model.
        """
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
        self._tokenizer.add_tokens(["<degC>", "<l_per_sqm>", "<kmh>", "<percent>", "<city>"], special_tokens=False)
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
