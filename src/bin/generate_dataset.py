#!/usr/group/env python3
import argparse
import re
import os
from typing import *
from collections import defaultdict

import numpy as np

from utils.random import set_seed


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
    if (
        len(args.splits) != 3
        or any(s < 0 or s > 100 for s in args.splits)
        or sum(args.splits) != 100
    ):
        raise ValueError(
            "The splits must be 3 comma-separated values that sum to 100: %s"
            % args.splits
        )


def normalize_args(args: Dict[str, Any]) -> None:
    args.input_path = os.path.realpath(args.input_path)
    args.output_path = os.path.realpath(args.output_path)
    args.splits = list(map(int, args.splits.split(",")))


def main(args: Dict[str, Any]) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    dataset_names = ["train", "dev", "test"]
    dataset_file = {}
    dataset_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for dataset_name in dataset_names:
        dataset_path = os.path.join(args.output_path, dataset_name + ".mk.tsv")
        dataset_file[dataset_name] = open(dataset_path, "w+")

    group_info = re.compile(r".*m_(\d+)\.t_(\d+)\.mk\.gp\.tsv")
    for proj in os.listdir(args.input_path):
        proj_base = os.path.join(args.input_path, proj)
        if not os.path.isdir(proj_base):
            continue
        for group in os.listdir(proj_base):
            group_path = os.path.join(proj_base, group)
            if not os.path.isfile(group_path):
                continue
            match = group_info.match(group_path)
            if match is None:
                continue
            ml, tl = match.groups()
            ml, tl = int(ml), int(tl)
            with open(group_path, "r") as file:
                nlines = sum(1 for line in file)
                indices = np.arange(nlines)
                splits = np.random.multinomial(
                    n=1, pvals=[s / 100 for s in args.splits], size=nlines
                ).argmax(axis=1)
                file.seek(0)
                for i, line in enumerate(file):
                    dataset_name = dataset_names[splits[i]]
                    dataset_file[dataset_name].write(line)
                    dataset_stats[dataset_name]["m"][ml] += 1
                    dataset_stats[dataset_name]["t"][tl] += 1

    for dataset_name in dataset_names:
        dataset_file[dataset_name].close()

    for dataset_name in dataset_names:
        masks = dataset_stats[dataset_name]["m"]
        masks = dict(sorted(masks.items(), key=lambda e: e[0]))
        print(dataset_name, "m", masks)

        tokens = dataset_stats[dataset_name]["t"]
        tokens = dict(sorted(tokens.items(), key=lambda e: e[0]))
        print(dataset_name, "t", tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="data/groups")
    parser.add_argument("--output-path", type=str, default="data/dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=str, default="60,10,30")
    args = parser.parse_args()

    normalize_args(args)
    validate_args(args)
    main(args)
