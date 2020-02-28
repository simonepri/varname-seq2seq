#!/usr/group/env python3
import argparse
import os
import re
import math
from typing import *  # pylint: disable=W0401,W0614

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
        if os.listdir(args.output_path):
            raise ValueError(
                "The output path is not empty: %s" % args.output_path
            )


def normalize_args(args: Dict[str, Any]) -> None:
    args.input_path = os.path.realpath(args.input_path)
    args.output_path = os.path.realpath(args.output_path)


def next_pow(exp: int, num: int) -> int:
    return int(math.pow(exp, math.ceil(math.log(num, exp))))


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
                with open(file_path, "r") as handle:
                    for line in handle:
                        masked_example = MaskedVarExample.deserialize(line)
                        src_num = len(masked_example.tokens)
                        trg_len = len(masked_example.target)
                        mask_num = len(masked_example.masked)
                        src_len_bucket = next_pow(2, src_num)
                        trg_len_bucket = next_pow(2, trg_len)
                        mask_num_bucket = next_pow(2, mask_num)

                        out_file = "s_%d.t_%d.m_%d.mk.gp.tsv" % (
                            src_len_bucket,
                            trg_len_bucket,
                            mask_num_bucket,
                        )
                        out_file_path = os.path.join(out_path, out_file)
                        with open(out_file_path, "a+") as group_file:
                            group_file.write(line)


if __name__ == "__main__":
    try:
        ARGS = parse_args()

        normalize_args(ARGS)
        validate_args(ARGS)
        main(ARGS)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
