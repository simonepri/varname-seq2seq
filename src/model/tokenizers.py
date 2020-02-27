from typing import *  # pylint: disable=W0401,W0614


class RobertaTokenizer:
    def __init__(self) -> None:
        # pylint: disable=C0415
        from transformers import RobertaTokenizer as Tokenizer

        self.tokenizer = Tokenizer.from_pretrained("roberta-base")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = len(self.tokenizer)

    # pylint: disable=R0201
    def name(self) -> str:
        return "roberta-bpe"

    def encode(self, text: str):
        return self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text)
        )

    def decode(self, sequence: List[int]):
        return self.tokenizer.decode(sequence)
