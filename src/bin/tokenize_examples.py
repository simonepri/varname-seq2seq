#!/usr/bin/env python3
import argparse
import os
import re
from typing import *

from transformers import RobertaTokenizer

from common.var_example import VarExample, TokenizedVarExample
from utils.files import walk_files, rebase_path
from utils.strings import rreplace


def validate_args(args: Dict[str, Any]) -> None:
    if not os.path.isdir(args.input_path):
        raise ValueError(
            "The input path must be a folder but it is not: %s"
            % args.input_path
        )
    if os.path.exists(args.output_path):
        if not os.path.isdir(args.output_path):
            raise ValueError(
                "The output path must be a folder: %s" % args.output_path
            )
        elif os.listdir(args.output_path):
            raise ValueError(
                "The output path is not empty: %s" % args.output_path
            )


def normalize_args(args: Dict[str, Any]) -> None:
    args.input_path = os.path.realpath(args.input_path)
    args.output_path = os.path.realpath(args.output_path)


def main(args: Dict[str, Any]) -> None:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    pattern = re.compile(r".*\.eg.tsv$")
    for path, files in walk_files(
        args.input_path, pattern, progress=True, batch=100
    ):
        out_path = rebase_path(args.input_path, args.output_path, path)
        os.makedirs(out_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(path, file)
            examples = VarExample.deserialize_from_file(file_path)
            tokenized_examples = TokenizedVarExample.from_var_examples(
                examples, tokenizer.tokenize
            )

            out_file = rreplace(".tsv", "", file) + ".tk.tsv"
            out_file_path = os.path.join(out_path, out_file)
            TokenizedVarExample.serialize_to_file(
                out_file_path, tokenized_examples
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="data/examples")
    parser.add_argument("--output-path", type=str, default="data/tokenized")
    args = parser.parse_args()

    normalize_args(args)
    validate_args(args)
    main(args)
