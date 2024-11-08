from typing import Dict

# so vom stil her wie the torchvision transforms wäre glaube ich ganz cool, also für jede einzelne Operation eine
# eigene Klasse ableiten

class RemoveCustomTokens:
    def __init__(self):
        pass

    def __call__(self, data: str) -> str:
        # TODO
        return data
