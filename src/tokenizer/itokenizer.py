from abc import ABC, abstractmethod
from typing import List


class ITokenizer(ABC):
    @abstractmethod
    def add_start_stop_tokens(self, s: str) -> str:
        ...
        
    @property
    @abstractmethod
    def padding_idx(self) -> int:
        ...
        
    @property
    @abstractmethod
    def start_idx(self) -> int:
        ...
    
    @property
    @abstractmethod
    def stop_idx(self) -> int:
        ...
    
    @property
    @abstractmethod
    def unknown_idx(self) -> int:
        ...
        
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...
    
    @abstractmethod
    def stoi(self, input: str) -> List[int]:
        ...
    
    @abstractmethod
    def itos(self, input: List[int]) -> str:        
        ...