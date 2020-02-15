import ast
from typing import *

from utils.strings import multiple_replace


class Serializable:
    @classmethod
    def serialize(cls, example: "Serializable") -> str:
        pass

    @classmethod
    def deserialize(cls, text: str) -> "Serializable":
        pass

    @classmethod
    def serialize_to_file(
        cls, file_path: str, examples: List["Serializable"]
    ) -> None:
        with open(file_path, "w+") as f:
            for example in examples:
                print(cls.serialize(example), file=f)

    @classmethod
    def deserialize_from_file(cls, file_path: str) -> List["Serializable"]:
        examples = []
        with open(file_path, "r") as f:
            for line in f:
                examples.append(cls.deserialize(line))
        return examples


class VarExample(Serializable):
    def __init__(self, tokens: List[str], masks: List[int]) -> None:
        assert len(tokens) == len(masks)
        self.tokens = tokens
        self.masks = masks

    def __iter__(self) -> Iterable[Tuple[str, int]]:
        return zip(self.tokens, self.masks)

    def __len__(self) -> int:
        return len(self.tokens)

    def variables(self) -> Set[int]:
        if hasattr(self, "vars"):
            return self.vars
        self.vars = set(self.masks)
        self.vars.remove(0)
        return self.vars

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
        tokens, masks, vars = [], [], set()
        parts = text.split("\t")
        for part in parts:
            token, _, varid = part.rpartition(":")
            token = cls.__decode_token(token)
            varid = int(varid)
            tokens.append(token)
            masks.append(varid)
            vars.add(varid)
        vars.remove(0)
        example = cls(tokens, masks)
        example.vars = vars
        return example

    @classmethod
    def __encode_token(cls, token: str) -> str:
        map = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
        return multiple_replace(map, token)

    @classmethod
    def __decode_token(cls, token: str) -> str:
        map = {r"\n": "\n", r"\r": "\r", r"\t": "\t"}
        return multiple_replace(map, token)


class MaskedVarExample(Serializable):
    def __init__(self, tokens: List[str], masked: List[int], target: str) -> None:
        self.tokens = tokens
        self.masked = masked
        self.target = target


    @classmethod
    def mask(cls, example: "VarExample", varid_to_mask: int) -> "MaskedVarExample":
        tokens, masked, target = [], [], None
        for i, (token, varid) in enumerate(example):
            if varid == varid_to_mask:
                tokens.append("*")
                masked.append(i)
                if target is None:
                    target = token
            else:
                tokens.append(token)
        return MaskedVarExample(tokens, masked, target)

    @classmethod
    def serialize(cls, masked_example: "MaskedVarExample") -> str:
        tokens = "\t".join(map(cls.__encode_token, masked_example.tokens))
        masked = ",".join(map(str, masked_example.masked))
        target = cls.__encode_token(masked_example.target)
        return "\t".join([tokens, masked, target])

    @classmethod
    def deserialize(cls, text: str) -> "MaskedVarExample":
        tokens, masked, target = text.rsplit("\t", 2)
        tokens = list(map(cls.__decode_token, tokens.split("\t")))
        masked = list(map(int, masked.split(",")))
        target = cls.__decode_token(target)
        return MaskedVarExample(tokens, masked, target)

    @classmethod
    def __encode_token(cls, token: str) -> str:
        map = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
        return multiple_replace(map, token)

    @classmethod
    def __decode_token(cls, token: str) -> str:
        map = {r"\n": "\n", r"\r": "\r", r"\t": "\t"}
        return multiple_replace(map, token)
