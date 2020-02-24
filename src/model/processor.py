import os
import pickle
from typing import *

import torch

from model.tokenizers import RobertaTokenizer
from model.config import Seq2SeqConfig


class Seq2SeqProcessor:
    def __init__(
        self, tokenizer, input_seq_max_length: int, output_seq_max_length: int
    ) -> None:
        self.tokenizer = tokenizer

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size

        self.input_seq_max_length = input_seq_max_length
        self.output_seq_max_length = output_seq_max_length

    def name(self) -> str:
        return "%s(%d,%d)" % (
            self.tokenizer.name(),
            self.input_seq_max_length,
            self.output_seq_max_length,
        )

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def decode(self, sequence: List[int]):
        return self.tokenizer.decode(sequence)

    def tensorise(
        self, source: List[str], target: str, masked: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(masked) > 0

        def trunc_to_len(list, max_length, include_pos=0):
            if len(list) > max_length:
                start_pos = max(0, include_pos - max_length // 2)
                end_pos = min(len(list), start_pos + max_length)
                start_pos = end_pos - max_length
                list = list[start_pos:end_pos]
            return list

        enc_source, enc_target, enc_masks = [], [], []

        enc_source.append(self.bos_token_id)
        masked_pos = set(masked)
        for pos, token in enumerate(source):
            if pos in masked_pos:
                enc_masks.append(len(enc_source))
                enc_source.append(self.mask_token_id)
                continue
            enc_source.extend(self.encode(token))
        enc_source.append(self.eos_token_id)
        enc_source = trunc_to_len(
            enc_source, self.input_seq_max_length, include_pos=enc_masks[-1]
        )

        enc_target.append(self.bos_token_id)
        enc_target.extend(self.encode(target))
        enc_target.append(self.eos_token_id)
        enc_target = trunc_to_len(enc_target, self.output_seq_max_length)

        enc_source = torch.tensor(enc_source, dtype=torch.int)
        enc_target = torch.tensor(enc_target, dtype=torch.int)

        assert len(enc_source) <= self.input_seq_max_length
        assert len(enc_target) <= self.output_seq_max_length

        return (enc_source, enc_target)

    @classmethod
    def from_config(cls, config: Seq2SeqConfig):
        if config.name == "roberta-bpe":
            tokenizer = RobertaTokenizer()
        else:
            raise NotImplementedError()
        return cls(
            tokenizer, config.input_seq_max_length, config.output_seq_max_length
        )
