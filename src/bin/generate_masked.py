#!/usr/bin/env python3
import argparse
import os
import re
from typing import *

from features.examples import VarExample, MaskedVarExample
from utils.files import walk_files, rebase_path
from utils.strings import rreplace

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default="data/examples")
    parser.add_argument("--output-path", type=str, default="data/masked")

    return parser.parse_args()


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
    pattern = re.compile(r".*\.eg.tsv$")
    for path, files in walk_files(
        args.input_path, pattern, progress=True, batch=100
    ):
        out_path = rebase_path(args.input_path, args.output_path, path)
        os.makedirs(out_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(path, file)

            examples = VarExample.deserialize_from_file(file_path)
            for i, example in enumerate(examples):
                out_file = rreplace(".eg.tsv", "", file) + ".%d.mk.tsv" % i
                out_file_path = os.path.join(out_path, out_file)
                with open(out_file_path, "w+") as f:
                    for varid_to_mask in example.variables():
                        masked_example = MaskedVarExample.mask(
                            example, varid_to_mask
                        )
                        masked_example_str = MaskedVarExample.serialize(
                            masked_example
                        )
                        print(masked_example_str, file=f)


if __name__ == "__main__":
    args = parse_args()

    normalize_args(args)
    validate_args(args)
    main(args)
