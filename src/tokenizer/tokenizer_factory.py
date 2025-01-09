#from .bert_tokenizer import BertTokenizer
from .bert_tokenizer_adapter import BertTokenizerAdapter
from .itokenizer import ITokenizer
from .set_of_words_tokenizer import SetOfWordsTokenizerDefault, SetOfWordsTokenizerGPT


class TokenizerFactory:
    def __init__(self):
        ...

    @staticmethod
    def get(dataset_path: str, tokenizer: str, target: str) -> ITokenizer:
        assert tokenizer in ["sow", "bert"], f"Unknown tokenizer '{tokenizer}'"
        assert target in ["default", "gpt"], f"Unknown target '{target}'"

        if tokenizer == "sow":
            if target == "default":
                return SetOfWordsTokenizerDefault(dataset_path)
            else:
                return SetOfWordsTokenizerGPT(dataset_path)
        else:
            return BertTokenizerAdapter()
