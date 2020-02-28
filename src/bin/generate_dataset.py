#!/usr/group/env python3
import argparse
import re
import os
import sys
from typing import *  # pylint: disable=W0401,W0614
from collections import defaultdict

import numpy as np
from prettytable import PrettyTable

from utils.random import set_seed


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default="data/groups")
    parser.add_argument("--output-path", type=str, default="data/dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=str, default="60,10,30")
    parser.add_argument("--no-splits", default=False, action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")

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
    if (
        (len(args.splits) != 3)
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
    args.include = [] if args.include == "" else args.include.split(",")
    args.exclude = [] if args.exclude == "" else args.exclude.split(",")


def main(args: Dict[str, Any]) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    ds_names = ["all"] if args.no_splits else ["train", "dev", "test"]
    if args.prefix != "":
        ds_names = list(map(lambda n: "%s.%s" % (args.prefix, n), ds_names))
    ds_file = {}
    ds_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for ds_name in ds_names:
        dataset_path = os.path.join(args.output_path, ds_name + ".mk.tsv")
        ds_file[ds_name] = open(dataset_path, "w+")

    pvals = [1.0] if args.no_splits else [s / 100.0 for s in args.splits]
    group_info = re.compile(r".*s_(\d+)\.t_(\d+)\.m_(\d+)\.mk\.gp\.tsv")
    projects = (
        os.listdir(args.input_path) if len(args.include) == 0 else args.include
    )
    for proj in projects:
        proj_base = os.path.join(args.input_path, proj)
        if proj in args.exclude or not os.path.isdir(proj_base):
            continue
        for group_file in os.listdir(proj_base):
            group_path = os.path.join(proj_base, group_file)
            if not os.path.isfile(group_path):
                continue
            match = group_info.match(group_path)
            if match is None:
                continue
            groups = match.groups()
            src_len_bucket = int(groups[0])
            trg_len_bucket = int(groups[1])
            mask_num_bucket = int(groups[2])
            with open(group_path, "r") as file:
                nlines = sum(1 for line in file)
                splits = np.random.multinomial(
                    n=1, pvals=pvals, size=nlines
                ).argmax(axis=1)
                file.seek(0)
                for i, line in enumerate(file):
                    ds_name = ds_names[splits[i]]
                    ds_file[ds_name].write(line)
                    ds_stats[ds_name]["s"][src_len_bucket] += 1
                    ds_stats[ds_name]["t"][trg_len_bucket] += 1
                    ds_stats[ds_name]["m"][mask_num_bucket] += 1

    for ds_name in ds_names:
        ds_file[ds_name].close()

    for metric in ["s", "t", "m"]:
        tab = PrettyTable()
        tab.field_names = [
            "# of %s" % metric,
            *["%s" % ds_name for ds_name in ds_names],
            "total",
        ]
        low, hig = sys.maxsize, 0
        for ds_name in ds_names:
            values = ds_stats[ds_name][metric].keys()
            low, hig = min(low, min(*values)), max(hig, max(*values))

        while low <= hig:
            values = [ds_stats[ds_name][metric][low] for ds_name in ds_names]
            total = sum(values)
            row = (
                "(%d, %d]" % (low // 2, low),
                *["%d" % v for v in values],
                "%d" % total,
            )
            tab.add_row(row)
            low *= 2
        print(tab)


if __name__ == "__main__":
    try:
        ARGS = parse_args()

        normalize_args(ARGS)
        validate_args(ARGS)
        main(ARGS)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
