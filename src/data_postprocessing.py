import re

from typing import Dict, List


class PostProcessingPipeline:
    def __init__(self, transforms: List):
        self._transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for transform in self._transforms:
            data = transform(data)

        return data


class ReplaceCityToken:
    """
    Replace the city token <city> with the correct city name.
    """
    def __init__(self):
        pass

    def __call__(self, s: str, city_name: str) -> str:
        return s.replace("<city>", city_name)
      

class RemovePunctutation:
    """
    Remove the most basic punctuation elements. The following elements are removed: . , ; ! ? :
    """
    def __init__(self):
        self._symbols = [".", ",", ";", "!", "?", ":"]

    def __call__(self, s: str) -> str:
        for symbol in self._symbols:
            s = s.replace(symbol, "")
            
        return s
    

class PostProcess:
    """
    Combine elementary post processing steps into one transformation. The following transformations are included:
    - RemoveRepeatedSpaces
    - CombineNegativeTemperatures
    - StripSpaces
    """
    def __init__(self):
        self._transforms = [
            RemoveRepeatedSpaces(),
            CombineNegativeTemperatures(),
            StripSpaces()
        ]

    def __call__(self, prediction: str) -> str:
        for transform in self._transforms:
            prediction = transform(prediction)
        
        return prediction 
    

class RemoveRepeatedSpaces:
    """
    Make sure that there are no repeated spaces.
    """
    def __init__(self):
        pass

    def __call__(self, s: str) -> str:
        return re.sub(r"[ ]{2,}", r" ", s)
    

class CombineNegativeTemperatures:
    """
    Make sure that the minus sign of a number is not separated by a space. This seems to happen from time to time with
    the bert tokenizer for unknown reasons.
    """
    def __init__(self):
        pass

    def __call__(self, s: str) -> str:
        return re.sub(r" (-) ([0-9]+)", " \1\2", s)
    

class StripSpaces:
    """
    Remove any leading and trailing spaces.
    """
    def __init__(self):
        pass

    def __call__(self, s: str) -> str:
        return s.strip()
