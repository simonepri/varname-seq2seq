#!/usr/bin/env python3
import argparse
import os
import re
from typing import *

from features.examples import VarExample
from features.java.ast import JavaAst
from features.java.extractor import JavaVarExamplesExtractor
from utils.files import walk_files, rebase_path


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default="data/corpora")
    parser.add_argument("--output-path", type=str, default="data/examples")
    parser.add_argument("--cache-only", default=False, action="store_true")

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
    JavaAst.setup()

    pattern = re.compile(r".*\.java$")
    for path, files in walk_files(
        args.input_path, pattern, progress=True, batch=100
    ):
        out_path = rebase_path(args.input_path, args.output_path, path)
        os.makedirs(out_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(path, file)
            if args.cache_only and not JavaAst.file_cached(file_path):
                continue
            try:
                examples = JavaVarExamplesExtractor.from_source_file(file_path)
            except Exception as e:
                print(end="", flush=True)
                print(e, flush=True)
                continue

            # Filter out examples that does not have any variable
            examples = list(filter(lambda e: len(e.variables()) > 0, examples))

            out_file = file + ".eg.tsv"
            out_file_path = os.path.join(out_path, out_file)
            VarExample.serialize_to_file(out_file_path, examples)


if __name__ == "__main__":
    try:
        args = parse_args()

        normalize_args(args)
        validate_args(args)
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
