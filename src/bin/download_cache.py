#!/usr/bin/env python3
import argparse
import os
import tarfile
from urllib.request import urlretrieve
from typing import *  # pylint: disable=W0401,W0614

from features.java.ast import JavaAst
from utils.progress import Progress, ByteProgress
from utils.strings import lreplace, rreplace


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default="data/corpora")
    parser.add_argument("--remove-prefix", type=str, default=":data:")

    return parser.parse_args()


def validate_args(args: Dict[str, Any]) -> None:
    if os.path.exists(JavaAst.AST_PROTO_DIR):
        if not os.path.isdir(JavaAst.AST_PROTO_DIR):
            raise ValueError(
                "The cache path must be a folder: %s" % JavaAst.AST_PROTO_DIR
            )
        if os.listdir(JavaAst.AST_PROTO_DIR):
            raise ValueError(
                "The cache path is not empty: %s" % JavaAst.AST_PROTO_DIR
            )


def normalize_args(args: Dict[str, Any]) -> None:
    args.data_path = os.path.realpath(args.data_path)


def main(
    args: Dict[str, Any]
) -> Iterable[Tuple[Tuple[str, List[str]], Tuple[str, List[str]]]]:
    remote_path = (
        "https://github.com/simonepri/varname-transformers"
        + "/releases/download/0.0.1/corpora-ast-cache.tgz"
    )
    destination_path = os.path.join(
        JavaAst.AST_PROTO_DIR, "corpora-ast-cache.tgz"
    )

    os.makedirs(JavaAst.AST_PROTO_DIR, exist_ok=True)
    with ByteProgress(desc=remote_path.split("/")[-1]) as pbar:
        urlretrieve(remote_path, destination_path, reporthook=pbar.update_to)

    with tarfile.open(destination_path, "r:gz") as tar:
        num_members = len(tar.getmembers())
        for member in Progress(iterable=tar.getmembers(), total=num_members):
            tar.extract(path=JavaAst.AST_PROTO_DIR, member=member)

    os.remove(destination_path)

    for file_name in os.listdir(JavaAst.AST_PROTO_DIR):
        file_path = os.path.join(JavaAst.AST_PROTO_DIR, file_name)
        if not os.path.isfile(file_path):
            continue
        if not file_name.endswith(".proto"):
            continue
        data_path = file_name
        data_path = lreplace(args.remove_prefix, "", data_path)
        data_path = rreplace(".proto", "", data_path)
        data_path = data_path.replace(":", os.sep)
        data_path = os.path.join(args.data_path, data_path)
        cache_path = JavaAst.cache_path_for_file(data_path)
        os.rename(file_path, cache_path)


if __name__ == "__main__":
    try:
        ARGS = parse_args()

        normalize_args(ARGS)
        validate_args(ARGS)
        main(ARGS)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
