#!/usr/bin/env python3
import argparse
import os
import re
from typing import *  # pylint: disable=W0401,W0614

from features.java.ast import JavaAst
from utils.files import walk_files


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default="data/corpora")

    return parser.parse_args()


def validate_args(args: Dict[str, Any]) -> None:
    if not os.path.exists(args.data_path):
        raise ValueError(
            "The data path provided does not exist: %s" % args.data_path
        )
    if not os.path.isdir(args.data_path):
        raise ValueError(
            "The data path provided is not a folder: %s" % args.data_path
        )


def normalize_args(args: Dict[str, Any]) -> None:
    args.data_path = os.path.realpath(args.data_path)


def main(args: Dict[str, Any]) -> None:
    JavaAst.setup()

    pattern = re.compile(r".*\.java$")
    for path, files in walk_files(args.data_path, pattern, progress=True):
        file_paths = list(map(lambda f, p=path: os.path.join(p, f), files))
        try:
            JavaAst.cache_files(file_paths)
        except IOError as err:
            print(flush=True, end="")
            print(err, flush=True)


if __name__ == "__main__":
    try:
        ARGS = parse_args()

        normalize_args(ARGS)
        validate_args(ARGS)
        main(ARGS)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
