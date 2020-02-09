#!/usr/bin/env python3
import argparse
import os
import re
import tarfile
from typing import *

from tqdm import tqdm

from utils.download import download_url


def validate_args(args: Dict[str, Any]) -> None:
    if os.path.exists(args.data_path):
        if not os.path.isdir(args.data_path):
            raise ValueError(
                "The data path must be a folder: %s" % args.data_path
            )
        elif os.listdir(args.data_path):
            raise ValueError("The data path is not empty: %s" % args.data_path)


def normalize_args(args: Dict[str, Any]) -> None:
    args.data_path = os.path.realpath(args.data_path)


def main(
    args: Dict[str, Any]
) -> Iterable[Tuple[Tuple[str, List[str]], Tuple[str, List[str]]]]:
    os.makedirs(args.data_path, exist_ok=True)

    remote_path = (
        "https://github.com/simonepri/varname-transformers"
        + "/releases/download/0.0.1/data.tgz"
    )
    destination_path = os.path.join(args.data_path, "corpus-sources.tgz")
    download_url(remote_path, destination_path, progress=True)

    with tarfile.open(destination_path, "r:gz") as tar:
        for member in tqdm(
            iterable=tar.getmembers(), total=len(tar.getmembers())
        ):
            tar.extract(path=args.data_path, member=member)

    os.remove(destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()

    validate_args(args)
    normalize_args(args)
    main(args)
