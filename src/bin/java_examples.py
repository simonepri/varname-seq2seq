#!/usr/bin/env python3
import argparse
import os
import re
from typing import *

from features.java.ast import JavaAst
from features.java.extractor import JavaLocalVarExamples
from utils.files import walk_files, split_file_path


def validate_args(args: Dict[str, Any]) -> None:
    if args.dir_mode and os.path.isfile(args.input_path):
        raise ValueError(
            "In dir mode the input path must be a folder but a file was given: %s"
            % args.input_path
        )
    if not args.dir_mode and os.path.isdir(args.input_path):
        raise ValueError(
            "The input path must be a file but a folder was given: %s"
            % args.input_path
        )


def normalize_args(args: Dict[str, Any]) -> None:
    args.input_path = os.path.realpath(args.input_path)
    args.output_path = os.path.realpath(args.output_path)


def get_data(
    args: Dict[str, Any], progress: bool = True
) -> Iterable[Tuple[Tuple[str, List[str]], Tuple[str, List[str]]]]:
    if not args.dir_mode:
        in_dir_path, in_file = split_file_path(args.input_path)
        out_dir_path, out_file = split_file_path(args.output_path)
        yield (in_dir_path, [in_file]), (out_dir_path, [out_file])
        return

    in_dir, out_dir = args.input_path, args.output_path
    pattern = re.compile(r".*\.java$")
    for in_dir_path, files in walk_files(in_dir, pattern, progress=progress):
        in_files, out_files = [], []

        in_rel_dir_path = os.path.relpath(in_dir_path, in_dir)
        in_rel_dir_flat = in_rel_dir_path.replace(os.sep, ":")

        out_dir_path = out_dir
        if not args.dir_flattening and in_rel_dir_path != ".":
            out_dir_path = os.path.join(out_dir_path, in_rel_dir_path)
        for in_file in files:
            out_file = in_file + ".eg.tsv"
            if args.dir_flattening and in_rel_dir_path != ".":
                out_file = in_rel_dir_flat + ":" + out_file
            in_files.append(in_file)
            out_files.append(out_file)
        yield (in_dir_path, in_files), (out_dir_path, out_files)


def main(args: Dict[str, Any]) -> None:
    JavaAst.setup(progress=True)

    for input, output in get_data(args, progress=True):
        (in_path, in_files), (out_path, out_files) = input, output
        for in_file, out_file in zip(in_files, out_files):
            in_file_path = os.path.join(in_path, in_file)
            if args.cache_only and not JavaAst.file_cached(in_file_path):
                continue
            try:
                examples = JavaLocalVarExamples.from_source_file(in_file_path)
            except Exception as e:
                print(flush=True, end="")
                print(e, flush=True)
                continue
            os.makedirs(out_path, exist_ok=True)
            out_file_path = os.path.join(out_path, out_file)
            examples.save(out_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--cache-only", default=False, action="store_true")
    parser.add_argument("--dir-mode", default=False, action="store_true")
    parser.add_argument("--dir-flattening", default=False, action="store_true")
    args = parser.parse_args()

    validate_args(args)
    normalize_args(args)
    main(args)
