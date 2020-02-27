#!/usr/bin/env python3
import argparse
import os
import re
from typing import *

from features.java.ast import JavaAst
from features.java.extractor import JavaLocalVarExamples
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
        file_paths = list(map(lambda f: os.path.join(path, f), files))
        try:
            JavaAst.cache_files(file_paths)
        except Exception as e:
            print(flush=True, end="")
            print(e, flush=True)


if __name__ == "__main__":
    try:
        args = parse_args()

        normalize_args(args)
        validate_args(args)
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
