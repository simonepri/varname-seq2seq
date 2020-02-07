#!/usr/bin/env python3
import argparse
import os
import re
from typing import *

from features.java.ast import JavaAst
from features.java.extractor import JavaLocalVarExamples
from utils.files import walk_files


def main(
    input_path: str,
    output_path: str,
    cache_only: bool,
    dir_mode: bool,
    dir_flatten: bool,
    precompute_cache: bool,
) -> None:
    input_path = os.path.relpath(input_path)
    output_path = os.path.relpath(output_path)

    JavaAst.setup(progress=True)

    if not dir_mode:
        if cache_only and not JavaAst.file_cached(input_path):
            return
        JavaLocalVarExamples.from_source_file(input_path).save(output_path)
        return

    os.makedirs(output_path, exist_ok=True)

    if precompute_cache:
        pattern = re.compile(r".*\.java$")
        for path, files in walk_files(input_path, pattern, progress=True):
            file_paths = list(map(lambda f: os.path.join(path, f), files))
            try:
                JavaAst.cache_files(file_paths)
            except Exception as e:
                print(flush=True, end="")
                print(e, flush=True)

    pattern = re.compile(r".*\.java$")
    for path, files in walk_files(input_path, pattern, progress=True):
        for file in files:
            in_file_path = os.path.join(path, file)
            if cache_only and not JavaAst.file_cached(in_file_path):
                continue
            if dir_flatten:
                out_file_path = in_file_path.replace("/", ":") + ".eg.tsv"
            else:
                os.makedirs(path, exist_ok=True)
            out_file_path = os.path.join(output_path, out_file_path + ".eg.tsv")
            try:
                JavaLocalVarExamples.from_source_file(in_file_path).save(
                    out_file_path
                )
            except Exception as e:
                print(flush=True, end="")
                print(e, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--cache-only", default=False, action="store_true")
    parser.add_argument("--dir", default=False, action="store_true")
    parser.add_argument("--flat", default=False, action="store_true")
    parser.add_argument("--precompute-cache", default=False, action="store_true")
    args = parser.parse_args()

    main(args.inp, args.out, args.cache_only, args.dir, args.flat, args.precompute_cache)
