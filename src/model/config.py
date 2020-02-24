import pickle
from typing import *


class Seq2SeqConfig(object):
    def __init__(self, **kwargs: Dict[str, Any]):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path: str) -> "Seq2SeqConfig":
        with open(file_path, "rb") as handle:
            return pickle.load(handle)
