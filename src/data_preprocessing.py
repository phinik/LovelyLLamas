from typing import Dict

# so vom stil her wie the torchvision transforms wäre glaube ich ganz cool, also für jede einzelne Operation eine
# eigene Klasse ableiten

class ReplaceLineBreaksInOverview:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        # TODO
        return data
    

class IntroduceCustomTokens:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        # TODO
        return data

    
class ReduceKeys:
    def __init__(self):
        pass

    def __call__(self, data: Dict) -> Dict:
        reduced_set_of_keys = ["city", "report_short", "overview"]

        reduced_dict = {}
        for key in reduced_set_of_keys:
            reduced_dict[key] = data[key]

        return data