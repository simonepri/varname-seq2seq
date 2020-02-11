#!/usr/group/env python3
import argparse
import os
import re
import math
from typing import *

from common.var_example import TokenizedVarExample
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
    pattern = re.compile(r".*\.eg.tk.tsv$")
    for file in os.listdir(args.input_path):
        proj_base = os.path.join(args.input_path, file)
        if not os.path.isdir(proj_base):
            continue

        out_path = rebase_path(args.input_path, args.output_path, proj_base)
        os.makedirs(out_path, exist_ok=True)
        group_files = {}
        for path, files in walk_files(
            proj_base, pattern, progress=True, batch=100
        ):
            for file in files:
                file_path = os.path.join(path, file)
                tokenized_examples = TokenizedVarExample.deserialize_from_file(
                    file_path
                )
                for tokenized_example in tokenized_examples:
                    text = TokenizedVarExample.serialize(tokenized_example)
                    size = tokenized_example.size()

                    # number of special tokens that will be added by RoBERTa
                    sp_size = 2
                    group_name = (
                        int(math.pow(2, math.ceil(math.log(size + sp_size, 2))))
                    )
                    if not group_name in group_files:
                        out_file = "%d.group.eg.tk.tsv" % group_name
                        out_file_path = os.path.join(out_path, out_file)
                        group_files[group_name] = open(out_file_path, "w+")
                    group_file = group_files[group_name]
                    print(text, file=group_file)

        for group in group_files:
            group_files[group].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="data/tokenized")
    parser.add_argument("--output-path", type=str, default="data/groups")
    args = parser.parse_args()

    normalize_args(args)
    validate_args(args)
    main(args)
