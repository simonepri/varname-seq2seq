#!/usr/group/env python3
import argparse
import os
from typing import *

import torch

from features.examples import MaskedVarExample
from features.java.ast import JavaAst
from features.java.extractor import JavaVarExamplesExtractor

from model.config import Seq2SeqConfig
from model.processor import Seq2SeqProcessor
from model.seq2seq import Seq2SeqModel


EXTRACTORS = {"java": JavaVarExamplesExtractor}

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str)
    parser.add_argument("--file-path", type=str)

    return parser.parse_args()

def validate_args(args: Dict[str, Any]) -> None:
    if extractor_for_file(args.file_path) is None:
        raise ValueError(
            "Currently only file with the following extesions are supported: %s"
            % list(EXTRACTORS.keys())
        )
    if not os.path.exists(args.file_path) or not os.path.isfile(args.file_path):
        raise ValueError(
            "The file path provided does not exists or is not a file: %s"
            % args.file_path
        )
    if not os.path.exists(args.model_path) or not os.path.isdir(
        args.model_path
    ):
        raise ValueError(
            "The model path provided does not exists or is not a folder: %s"
            % args.model_path
        )


def normalize_args(args: Dict[str, Any]) -> None:
    args.file_path = os.path.realpath(args.file_path)
    args.model_path = os.path.realpath(args.model_path)


def extractor_for_file(file_path):
    extension = ""
    parts = file_path.rsplit(".", 1)
    if len(parts) == 2:
        extension = parts[1].lower()
    return EXTRACTORS.get(extension, None)


def main(args: Dict[str, Any]) -> None:
    JavaAst.setup(progress=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config_file = os.path.join(args.model_path, "config.pkl")
    model_state_file = os.path.join(args.model_path, "model.pt")

    config = Seq2SeqConfig.load(model_config_file)
    processor = Seq2SeqProcessor.from_config(config.processor)
    model = Seq2SeqModel.from_config(config.model).to(device)
    model.load_state_dict(torch.load(model_state_file, map_location=device))

    extractor = extractor_for_file(args.file_path)
    examples = extractor.from_source_file(args.file_path)

    for i, example in enumerate(examples):
        if len(example.variables()) == 0:
            return

        for j, varid_to_mask in enumerate(example.variables()):
            masked_example = MaskedVarExample.mask(example, varid_to_mask)
            source, target = processor.tensorise(
                masked_example.tokens,
                masked_example.target,
                masked_example.masked,
            )
            source = source.long().to(device)
            prediction = model.run_prediction(
                source, processor.output_seq_max_length
            )

            target = target[1:]
            prediction = prediction[1:]
            if target[-1] == processor.eos_token_id:
                target = target[:-1]
            if prediction[-1] == processor.eos_token_id:
                prediction = prediction[:-1]

            target = processor.decode(target)
            prediction = processor.decode(prediction)
            if prediction == target:
                print("✔️  | %03d.%03d | %s" % (i + 1, j + 1, target))
                continue
            print(
                "⚠️  | %03d.%03d | %s → %s" % (i + 1, j + 1, target, prediction)
            )


if __name__ == "__main__":
    try:
        args = parse_args()

        normalize_args(args)
        validate_args(args)
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
