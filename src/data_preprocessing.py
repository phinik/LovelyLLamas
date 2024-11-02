from abc import ABC, abstractmethod
from typing import Dict

# so vom stil her wie the torchvision transforms wäre glaube ich ganz cool, also für jede einzelne Operation eine
# eigene Klasse ableiten

class ITransform(ABC):
    @abstractmethod
    def process(self, data: Dict) -> Dict:
        ...


class ReplaceLineBreaksInOverview(ITransform):
    def __init__(self):
        pass

    def process(self, data: Dict) -> Dict:
        # TODO
        return data
    

class IntroduceCustomTokens(ITransform):
    def __init__(self):
        pass

    def process(self, data: Dict) -> Dict:
        # TODO
        return data

    