#!/usr/group/env python3
import argparse
import sys
import os
import time
import math
import pickle
from typing import *

import torch
import numpy as np
from tqdm import tqdm

from features.examples import MaskedVarExample
from model.data import Seq2SeqDataset, Seq2SeqDataLoader
from model.seq2seq import Seq2SeqModel
from model.processor import Seq2SeqProcessor
from utils.random import set_seed
from utils.configs import Config


def validate_args(args: Dict[str, Any]) -> None:
    if not args.do_train and not args.do_test:
        raise ValueError("Nothing to do")
    if args.do_train and args.epochs < 1:
        raise ValueError("The number of epoch must be positive")
    if os.path.exists(args.output_path):
        if not os.path.isdir(args.output_path):
            raise ValueError(
                "The output path must be a folder: %s" % args.output_path
            )
    if args.do_train and not os.path.isfile(args.train_file):
        raise ValueError(
            "The training file does not exist: %s" % args.train_file
        )
    if args.do_train and not os.path.isfile(args.dev_file):
        raise ValueError(
            "The development file does not exist: %s" % args.dev_file
        )
    if args.do_test and not os.path.isfile(args.test_file):
        raise ValueError("The test file does not exist: %s" % args.test_file)


def normalize_args(args: Dict[str, Any]) -> None:
    args.train_file = os.path.realpath(args.train_file)
    args.dev_file = os.path.realpath(args.dev_file)
    args.test_file = os.path.realpath(args.test_file)
    args.output_path = os.path.realpath(args.output_path)


def load_and_cache_data(
    processor: Seq2SeqProcessor, dataset_path: str, output_path: str,
) -> None:
    _, filename = os.path.split(dataset_path)
    cached_file_name = "%s.%s.cache.pkl" % (filename, processor.name())
    cached_file_path = os.path.join(output_path, cached_file_name)

    if os.path.exists(cached_file_path):
        with open(cached_file_path, "rb") as handle:
            data = pickle.load(handle)
    else:
        examples = MaskedVarExample.deserialize_from_file(dataset_path)
        data = [
            processor.tensorise(example.tokens, example.target, example.masked)
            for example in tqdm(examples)
        ]
        with open(cached_file_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def main(args: Dict[str, Any]) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor_config = Config(
        processor_name=args.processor_name,
        input_seq_max_length=args.input_seq_max_length,
        output_seq_max_length=args.output_seq_max_length,
    )
    processor = Seq2SeqProcessor.from_config(processor_config)

    model_config = Config(
        encoder_name=args.encoder_name,
        decoder_name=args.decoder_name,
        vocab_size=processor.vocab_size,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        input_seq_max_length=args.input_seq_max_length,
        output_seq_max_length=args.output_seq_max_length,
        bos_token_id=processor.bos_token_id,
        eos_token_id=processor.eos_token_id,
        pad_token_id=processor.pad_token_id,
        mask_token_id=processor.mask_token_id,
        num_layers=args.num_layers,
        layers_dropout=0.5,
        embedding_dropout=0.5,
    )
    model = Seq2SeqModel.from_config(model_config).to(device)

    model_filename = "best-%s.pt" % model.name()
    model_output_path = os.path.join(args.output_path, model_filename)

    optimizer = torch.optim.Adam(model.parameters())

    best_valid_loss = float("inf")

    if args.do_train:
        print(
            f"The model has {model.count_parameters():,} trainable parameters"
        )

        print("Loading the training dataset")
        train_data = load_and_cache_data(
            processor, args.train_file, args.output_path
        )
        train_dataset = Seq2SeqDataset(train_data)

        print("Loading the validation dataset")
        valid_data = load_and_cache_data(
            processor, args.dev_file, args.output_path
        )
        valid_dataset = Seq2SeqDataset(valid_data)

        epoch_iterator = range(1, args.epochs + 1)
        for epoch in epoch_iterator:
            print("¤ Epoch %d / %d" % (epoch, args.epochs))

            train_it = Seq2SeqDataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                device=device,
            )
            train_it = tqdm(train_it, desc="├ Train", file=sys.stdout)
            train_loss, train_acc = model.run_epoch(
                train_it, optimizer=optimizer,
            )
            print(
                "│ └ Loss: %.3f | Acc: %.2f%%" % (train_loss, train_acc * 100)
            )

            valid_it = Seq2SeqDataLoader(
                valid_dataset, batch_size=args.batch_size, device=device
            )
            valid_it = tqdm(valid_it, desc="└ Valid", file=sys.stdout)
            valid_loss, valid_acc = model.run_epoch(valid_it)
            if valid_loss >= best_valid_loss or math.isinf(best_valid_loss):
                print(
                    "  └ Loss: %.3f | Acc: %.2f%%"
                    % (valid_loss, valid_acc * 100)
                )
            else:
                delta_loss = valid_loss - best_valid_loss
                print(
                    "  └ Loss: %.3f (%.3fΔ) | Acc: %.2f%%"
                    % (valid_loss, delta_loss, valid_acc * 100)
                )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_output_path)

    if args.do_test:
        if os.path.exists(model_output_path):
            model.load_state_dict(torch.load(model_output_path))

        print("Loading the test dataset")
        test_data = load_and_cache_data(
            processor, args.test_file, args.output_path
        )
        test_dataset = Seq2SeqDataset(test_data)

        print("¤ Evaluation ")
        test_it = Seq2SeqDataLoader(
            test_dataset, batch_size=args.batch_size, device=device
        )
        test_it = tqdm(test_it, desc="├ Test", file=sys.stdout)

        test_loss, test_acc = model.run_epoch(test_it)
        print("  └ Loss: %.3f | Acc: %.3f%%" % (test_loss, test_acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file", type=str, default="data/dataset/train.mk.tsv"
    )
    parser.add_argument(
        "--dev-file", type=str, default="data/dataset/dev.mk.tsv"
    )
    parser.add_argument(
        "--test-file", type=str, default="data/dataset/test.mk.tsv"
    )
    parser.add_argument("--output-path", type=str, default="data/models")
    parser.add_argument("--processor-name", type=str, default="roberta")
    parser.add_argument("--encoder-name", type=str, default="lstm")
    parser.add_argument("--decoder-name", type=str, default="lstm")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--input-seq-max-length", type=int, default=256)
    parser.add_argument("--output-seq-max-length", type=int, default=64)
    parser.add_argument("--do-train", default=False, action="store_true")
    parser.add_argument("--do-test", default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    normalize_args(args)
    validate_args(args)
    main(args)
