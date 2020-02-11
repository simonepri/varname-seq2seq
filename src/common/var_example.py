from typing import *

from utils.strings import multiple_replace


class VarExample:
    def __init__(self, tokens: List[str], masks: List[int]) -> None:
        assert len(tokens) == len(masks)
        self.tokens = tokens
        self.masks = masks

    def __iter__(self) -> Iterable[Tuple[str, int]]:
        return zip(self.tokens, self.masks)

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def serialize(cls, example: "VarExample") -> str:
        example_builder = []
        for token, varid in example:
            token = cls.__encode_token(token)
            example_builder.append("%s:%d" % (token, varid))
        example_str = "\t".join(example_builder)
        return example_str

    @classmethod
    def deserialize(cls, text: str) -> "VarExample":
        tokens, masks = [], []
        parts = line.split("\t")
        for part in parts:
            token, _, varid = part.rpartition(":")
            token = cls.__decode_token(token)
            tokens.append(token)
            masks.append(varid)
        return cls(tokens, masks)

    @classmethod
    def serialize_to_file(
        cls, file_path: str, examples: List["VarExample"]
    ) -> None:
        with open(file_path, "w+") as f:
            for example in examples:
                print(cls.serialize(example), file=f)

    @classmethod
    def deserialize_from_file(cls, file_path: str) -> List["VarExample"]:
        examples = []
        with open(file_path, "r") as f:
            for line in f:
                examples.append(cls.deserialize(line))
        return examples

    @classmethod
    def __encode_token(cls, token: str) -> str:
        map = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
        return multiple_replace(map, token)

    @classmethod
    def __decode_token(cls, token: str) -> str:
        map = {r"\n": "\n", r"\r": "\r", r"\t": "\t"}
        return multiple_replace(map, token)
