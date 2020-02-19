from typing import *


class Config(object):
    def __init__(self, **kwargs: Dict[str, Any]):
        # Additional attributes without default values
        for key, value in kwargs.items():
            setattr(self, key, value)
