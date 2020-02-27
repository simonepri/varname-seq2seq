#!/usr/bin/env python3
import argparse
import os
import tarfile
from urllib.request import urlretrieve
from typing import *  # pylint: disable=W0401,W0614

from utils.progress import Progress, ByteProgress


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--file-name", type=str)
    parser.add_argument("--data-path", type=str)

    return parser.parse_args()


def validate_args(args: Dict[str, Any]) -> None:
    if os.path.exists(args.data_path):
        if not os.path.isdir(args.data_path):
            raise ValueError(
                "The data path must be a folder: %s" % args.data_path
            )
        if os.listdir(args.data_path):
            raise ValueError("The data path is not empty: %s" % args.data_path)


def normalize_args(args: Dict[str, Any]) -> None:
    args.data_path = os.path.realpath(args.data_path)


def main(
    args: Dict[str, Any]
) -> Iterable[Tuple[Tuple[str, List[str]], Tuple[str, List[str]]]]:
    os.makedirs(args.data_path, exist_ok=True)

    remote_path = (
        "https://github.com/simonepri/varname-transformers"
        + "/releases/download/0.0.1/"
        + args.file_name
    )
    destination_path = os.path.join(args.data_path, args.file_name)
    with ByteProgress(desc=remote_path.split("/")[-1]) as pbar:
        urlretrieve(remote_path, destination_path, reporthook=pbar.update_to)

    with tarfile.open(destination_path, "r:gz") as tar:
        num_members = len(tar.getmembers())
        for member in Progress(iterable=tar.getmembers(), total=num_members):
            tar.extract(path=args.data_path, member=member)

    os.remove(destination_path)


if __name__ == "__main__":
    try:
        ARGS = parse_args()

        normalize_args(ARGS)
        validate_args(ARGS)
        main(ARGS)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
