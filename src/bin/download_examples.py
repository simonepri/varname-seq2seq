#!/usr/bin/env python3
import argparse
import os
import tarfile
from typing import *

from tqdm import tqdm

from utils.download import download_url


def validate_args(args: Dict[str, Any]) -> None:
    if os.path.exists(args.examples_path):
        if not os.path.isdir(args.examples_path):
            raise ValueError(
                "The examples path must be a folder: %s" % args.examples_path
            )
        elif os.listdir(args.examples_path):
            raise ValueError(
                "The examples path is not empty: %s" % args.examples_path
            )


def normalize_args(args: Dict[str, Any]) -> None:
    args.examples_path = os.path.realpath(args.examples_path)


def main(
    args: Dict[str, Any]
) -> Iterable[Tuple[Tuple[str, List[str]], Tuple[str, List[str]]]]:
    os.makedirs(args.examples_path, exist_ok=True)

    remote_path = (
        "https://github.com/simonepri/varname-transformers"
        + "/releases/download/0.0.1/corpora-examples.tgz"
    )
    destination_path = os.path.join(args.examples_path, "corpora-examples.tgz")
    download_url(remote_path, destination_path, progress=True)

    with tarfile.open(destination_path, "r:gz") as tar:
        for member in tqdm(
            iterable=tar.getmembers(), total=len(tar.getmembers())
        ):
            tar.extract(path=args.examples_path, member=member)

    os.remove(destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-path", type=str, default="data/examples")
    args = parser.parse_args()

    validate_args(args)
    normalize_args(args)
    main(args)
