#!/usr/bin/env python3
import argparse
import os
import re
import tarfile
from typing import *

from tqdm import tqdm

from utils.download import download_url


def validate_args(args: Dict[str, Any]) -> None:
    if os.path.exists(args.cache_path):
        if not os.path.isdir(args.cache_path):
            raise ValueError(
                "The cache path must be a folder: %s" % args.cache_path
            )
        elif os.listdir(args.cache_path):
            raise ValueError(
                "The cache path is not empty: %s" % args.cache_path
            )


def normalize_args(args: Dict[str, Any]) -> None:
    args.data_path = os.path.realpath(args.data_path)
    args.cache_path = os.path.realpath(args.cache_path)


def main(
    args: Dict[str, Any]
) -> Iterable[Tuple[Tuple[str, List[str]], Tuple[str, List[str]]]]:
    os.makedirs(args.cache_path, exist_ok=True)

    remote_path = (
        "https://github.com/simonepri/varname-transformers"
        + "/releases/download/0.0.1/corpora-ast-cache.tgz"
    )
    destination_path = os.path.join(args.cache_path, "corpora-ast-cache.tgz")
    download_url(remote_path, destination_path, progress=True)

    with tarfile.open(destination_path, "r:gz") as tar:
        for member in tqdm(
            iterable=tar.getmembers(), total=len(tar.getmembers())
        ):
            tar.extract(path=args.cache_path, member=member)

    os.remove(destination_path)

    for file_name in os.listdir(args.cache_path):
        file_path = os.path.join(args.cache_path, file_name)
        if not os.path.isfile(file_path):
            continue
        new_file_name = file_name
        if args.remove_prefix != "":
            new_file_name = new_file_name.replace(args.remove_prefix, "")
        if args.data_path != "":
            source_path = new_file_name.replace(":", os.sep)
            new_source_path = os.path.join(args.data_path, source_path)
            new_file_name = new_source_path.replace(os.sep, ":")
        new_file_path = os.path.join(args.cache_path, new_file_name)
        print(file_path, new_file_path)
        os.rename(file_path, new_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/corpora")
    parser.add_argument("--remove-prefix", type=str, default=":data:")
    parser.add_argument(
        "--cache-path", type=str, default=".cache/java_ast/proto"
    )
    args = parser.parse_args()

    validate_args(args)
    normalize_args(args)
    main(args)
