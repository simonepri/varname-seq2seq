import ast
from typing import *

from transformers import PreTrainedTokenizer

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
        parts = text.split("\t")
        for part in parts:
            token, _, varid = part.rpartition(":")
            token = cls.__decode_token(token)
            varid = int(varid)
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


class TokenizedVarExample:
    def __init__(self, multi_tokens: List[List[str]], masks: List[int]) -> None:
        assert len(multi_tokens) == len(masks)
        self.multi_tokens = multi_tokens
        self.masks = masks

    def __iter__(self) -> Iterable[Tuple[List[str], int]]:
        return zip(self.multi_tokens, self.masks)

    def __len__(self) -> int:
        return len(self.multi_tokens)

    def size(self) -> int:
        if hasattr(self, 'tlen'):
            return self.tlen
        self.tlen = 0
        for multi_token in self.multi_tokens:
            self.tlen += len(multi_token)
        return self.tlen

    @classmethod
    def from_var_example(
        cls, example: VarExample, tokenizer: Callable[[str], List[str]]
    ) -> "TokenizedVarExample":
        tlen = 0
        multi_tokens = []
        for token, varid in example:
            multi_token = tokenizer(token)
            multi_tokens.append(multi_token)
            tlen += len(multi_token)
        tokenized_example = cls(multi_tokens, example.masks)
        tokenized_example.tlen = tlen
        return tokenized_example

    @classmethod
    def from_var_examples(
        cls, examples: List[VarExample], tokenizer: Callable[[str], List[str]]
    ) -> List["TokenizedVarExample"]:
        tokenized_examples = []
        for example in examples:
            tokenized_example = cls.from_var_example(example, tokenizer)
            tokenized_examples.append(tokenized_example)
        return tokenized_examples

    @classmethod
    def serialize(cls, tokenized_example: "TokenizedVarExample") -> str:
        tokens = []
        for multi_token, _ in tokenized_example:
            token = "\t".join(multi_token)
            tokens.append(token)
        example = VarExample(tokens, tokenized_example.masks)
        return VarExample.serialize(example)

    @classmethod
    def deserialize(cls, text: str) -> "TokenizedVarExample":
        example = VarExample.deserialize(text)
        tlen = 0
        multi_tokens = []
        for token, _ in example:
            multi_token = token.split("\t")
            multi_tokens.append(multi_token)
            tlen += len(multi_token)
        tokenized_example = cls(multi_tokens, example.masks)
        tokenized_example.tlen = tlen
        return tokenized_example

    @classmethod
    def serialize_to_file(
        cls, file_path: str, examples: List["TokenizedVarExample"]
    ) -> None:
        with open(file_path, "w+") as f:
            for example in examples:
                print(cls.serialize(example), file=f)

    @classmethod
    def deserialize_from_file(
        cls, file_path: str
    ) -> List["TokenizedVarExample"]:
        examples = []
        with open(file_path, "r") as f:
            for line in f:
                examples.append(cls.deserialize(line))
        return examples
