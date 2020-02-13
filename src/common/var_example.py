import ast
from typing import *

from transformers import PreTrainedTokenizer

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
                masked.append(i + 1)
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


class TokenizedVarExample(VarExample):
    def __init__(self, multi_tokens: List[List[str]], masks: List[int]) -> None:
        assert len(multi_tokens) == len(masks)
        self.multi_tokens = multi_tokens
        self.masks = masks

    def __iter__(self) -> Iterable[Tuple[List[str], int]]:
        return zip(self.multi_tokens, self.masks)

    def __len__(self) -> int:
        return len(self.multi_tokens)

    def variables(self) -> Set[int]:
        return super(TokenizedVarExample, self).variables()

    def size(self) -> int:
        if hasattr(self, "tlen"):
            return self.tlen
        self.tlen = 0
        for multi_token in self.multi_tokens:
            self.tlen += len(multi_token)
        return self.tlen

    @classmethod
    def serialize(cls, tokenized_example: "TokenizedVarExample") -> str:
        example = cls.to_var_example(
            tokenized_example, lambda mt: "\t".join(mt)
        )
        return VarExample.serialize(example)

    @classmethod
    def deserialize(cls, text: str) -> "TokenizedVarExample":
        example = VarExample.deserialize(text)
        return cls.from_var_example(example, lambda t: t.split("\t"))

    @classmethod
    def to_var_example(
        cls,
        tokenized_example: "TokenizedVarExample",
        untokenizer: Callable[[List[str]], str],
    ) -> VarExample:
        tokens = []
        for multi_token, _ in tokenized_example:
            token = untokenizer(multi_token)
            tokens.append(token)
        example = VarExample(tokens, tokenized_example.masks)
        example.vars = tokenized_example.vars
        return example

        tlen = 0
        multi_tokens = []
        for token, _ in example:
            multi_token = tokenizer(token)
            multi_tokens.append(multi_token)
            tlen += len(multi_token)
        tokenized_example = cls(multi_tokens, example.masks)
        tokenized_example.tlen = tlen
        tokenized_example.vars = example.vars
        return tokenized_example

    @classmethod
    def from_var_example(
        cls, example: VarExample, tokenizer: Callable[[str], List[str]]
    ) -> "TokenizedVarExample":
        tlen = 0
        multi_tokens = []
        for token, _ in example:
            multi_token = tokenizer(token)
            multi_tokens.append(multi_token)
            tlen += len(multi_token)
        tokenized_example = cls(multi_tokens, example.masks)
        tokenized_example.tlen = tlen
        tokenized_example.vars = example.vars
        return tokenized_example

    @classmethod
    def to_var_examples(
        cls,
        examples: List["TokenizedVarExample"],
        tokenizer: Callable[[str], List[str]],
    ) -> List[VarExample]:
        tokenized_examples = []
        for example in examples:
            tokenized_example = cls.to_var_example(example, tokenizer)
            tokenized_examples.append(tokenized_example)
        return tokenized_examples

    @classmethod
    def from_var_examples(
        cls, examples: List[VarExample], tokenizer: Callable[[str], List[str]]
    ) -> List["TokenizedVarExample"]:
        tokenized_examples = []
        for example in examples:
            tokenized_example = cls.from_var_example(example, tokenizer)
            tokenized_examples.append(tokenized_example)
        return tokenized_examples
