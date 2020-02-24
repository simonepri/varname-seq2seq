#!/usr/group/env python3
import argparse
import os
import re
import math
from typing import *

from features.examples import MaskedVarExample
from utils.files import walk_files, rebase_path

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default="data/masked")
    parser.add_argument("--output-path", type=str, default="data/groups")

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
    pattern = re.compile(r".*\.mk.tsv$")

    for proj in os.listdir(args.input_path):
        proj_base = os.path.join(args.input_path, proj)
        if not os.path.isdir(proj_base):
            continue

        out_path = rebase_path(args.input_path, args.output_path, proj_base)
        os.makedirs(out_path, exist_ok=True)
        for path, files in walk_files(
            proj_base, pattern, progress=True, batch=100
        ):
            for file in files:
                file_path = os.path.join(path, file)
                with open(file_path, "r") as f:
                    for line in f:
                        masked_example = MaskedVarExample.deserialize(line)
                        mask_num = len(masked_example.masked)
                        token_num = len(masked_example.tokens)
                        ml = int(math.pow(2, math.ceil(math.log(mask_num, 2))))
                        tl = int(math.pow(2, math.ceil(math.log(token_num, 2))))

                        out_file = "m_%d.t_%d.mk.gp.tsv" % (ml, tl)
                        out_file_path = os.path.join(out_path, out_file)
                        with open(out_file_path, "a+") as group_file:
                            group_file.write(line)


if __name__ == "__main__":
    try:
        args = parse_args()

        normalize_args(args)
        validate_args(args)
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
