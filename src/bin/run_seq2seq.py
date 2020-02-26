#!/usr/group/env python3
import argparse
import sys
import os
import time
import math
import pickle
from typing import *

import torch
from tqdm import tqdm

from features.examples import MaskedVarExample
from model.config import Seq2SeqConfig
from model.data import Seq2SeqDataset, Seq2SeqDataLoader
from model.processor import Seq2SeqProcessor
from model.seq2seq import Seq2SeqModel
from utils.random import set_seed


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-file", type=str, default="data/dataset/train.mk.tsv"
    )
    parser.add_argument(
        "--valid-file", type=str, default="data/dataset/dev.mk.tsv"
    )
    parser.add_argument(
        "--test-file", type=str, default="data/dataset/test.mk.tsv"
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--cache-path", type=str, default=".cache/model")
    parser.add_argument("--output-path", type=str, default="data/models")

    parser.add_argument("--processor-name", type=str, default="roberta-bpe")
    parser.add_argument("--encoder-name", type=str, default="rnn")
    parser.add_argument("--decoder-name", type=str, default="rnn")

    parser.add_argument("--rnn-cell", type=str, default="lstm")
    parser.add_argument("--rnn-num-layers", type=int, default=2)
    parser.add_argument("--rnn-hidden-size", type=int, default=512)
    parser.add_argument("--rnn-layers-dropout", type=float, default=0.5)
    parser.add_argument("--rnn-embedding-size", type=int, default=256)
    parser.add_argument("--rnn-embedding-dropout", type=float, default=0.5)

    parser.add_argument("--input-seq-max-length", type=int, default=256)
    parser.add_argument("--output-seq-max-length", type=int, default=64)

    parser.add_argument("--do-train", default=False, action="store_true")
    parser.add_argument("--do-test", default=False, action="store_true")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def validate_args(args: Dict[str, Any]) -> None:
    if not args.do_train and not args.do_test:
        raise ValueError("Nothing to do")
    if args.run_id == "":
        raise ValueError("You must specify a run id")
    if args.do_train and args.epochs < 0:
        raise ValueError("The number of epoch must be non negative")
    if os.path.exists(args.output_path):
        if not os.path.isdir(args.output_path):
            raise ValueError(
                "The output path must be a folder: %s" % args.output_path
            )
    if os.path.exists(args.cache_path):
        if not os.path.isdir(args.cache_path):
            raise ValueError(
                "The cache path must be a folder: %s" % args.cache_path
            )
    if args.do_train:
        if not os.path.isfile(args.train_file):
            raise ValueError(
                "The training file does not exist: %s" % args.train_file
            )
        if not os.path.isfile(args.valid_file):
            raise ValueError(
                "The validation file does not exist: %s" % args.valid_file
            )
    if args.do_test:
        if not os.path.isfile(args.test_file):
            raise ValueError(
                "The test file does not exist: %s" % args.test_file
            )
        if not args.do_train and not os.path.exists(
            os.path.join(args.output_path, args.run_id)
        ):
            raise ValueError(
                "Output folder for the run id provided not found: %s"
                % args.run_id
            )


def normalize_args(args: Dict[str, Any]) -> None:
    args.train_file = os.path.realpath(args.train_file)
    args.valid_file = os.path.realpath(args.valid_file)
    args.test_file = os.path.realpath(args.test_file)
    args.output_path = os.path.realpath(args.output_path)
    args.cache_path = os.path.realpath(args.cache_path)

    if args.run_id == "" and args.do_train:
        args.run_id = time.strftime("%Y-%m-%d-%H-%M-%S")


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


def build_processor_config(args: Dict[str, Any]) -> Seq2SeqConfig:
    if args.processor_name == "roberta-bpe":
        processor_config = Seq2SeqConfig(
            name=args.processor_name,
            input_seq_max_length=args.input_seq_max_length,
            output_seq_max_length=args.output_seq_max_length,
        )
    else:
        raise NotImplementedError()
    return processor_config


def build_model_config(
    processor: Seq2SeqProcessor, args: Dict[str, Any]
) -> Seq2SeqConfig:
    if args.encoder_name == "rnn":
        encoder_config = Seq2SeqConfig(
            name=args.encoder_name,
            vocab_size=processor.vocab_size,
            rnn_cell=args.rnn_cell,
            hidden_size=args.rnn_hidden_size,
            embedding_size=args.rnn_embedding_size,
            num_layers=args.rnn_num_layers,
            layers_dropout=args.rnn_layers_dropout,
            embedding_dropout=args.rnn_embedding_dropout,
        )
    else:
        raise NotImplementedError()

    if args.decoder_name == "rnn":
        decoder_config = Seq2SeqConfig(
            name=args.decoder_name,
            vocab_size=processor.vocab_size,
            rnn_cell=args.rnn_cell,
            hidden_size=args.rnn_hidden_size,
            embedding_size=args.rnn_embedding_size,
            num_layers=args.rnn_num_layers,
            layers_dropout=args.rnn_layers_dropout,
            embedding_dropout=args.rnn_embedding_dropout,
        )
    else:
        raise NotImplementedError()

    model_config = Seq2SeqConfig(
        encoder=encoder_config,
        decoder=decoder_config,
        input_seq_max_length=args.input_seq_max_length,
        output_seq_max_length=args.output_seq_max_length,
        bos_token_id=processor.bos_token_id,
        eos_token_id=processor.eos_token_id,
        pad_token_id=processor.pad_token_id,
        mask_token_id=processor.mask_token_id,
    )
    return model_config


def metrics_str(metrics: Dict[str, float]) -> str:
    pieces = []
    for key in metrics:
        pieces.append("%s: %.4f%%" % (key.capitalize(), metrics[key]))
    return " | ".join(pieces)


def train(
    run_id: str,
    epochs: int,
    batch_size: int,
    cache_path: str,
    output_path: str,
    train_file: str,
    valid_file: str,
) -> None:
    print("Starting training with run id %s" % run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor_config = build_processor_config(args)
    processor = Seq2SeqProcessor.from_config(processor_config)
    model_config = build_model_config(processor, args)
    model = Seq2SeqModel.from_config(model_config).to(device)
    config = Seq2SeqConfig(processor=processor_config, model=model_config)

    run_path = os.path.join(output_path, run_id)
    os.makedirs(run_path, exist_ok=False)

    model_file_path = os.path.join(run_path, "model.pt")
    config_file_path = os.path.join(run_path, "config.pkl")
    metrics_file_path = os.path.join(run_path, "metrics.tsv")

    print("The model has %d trainable parameters" % model.count_parameters())

    print("Saving initial model and configs")
    torch.save(model.state_dict(), model_file_path)
    config.save(config_file_path)
    with open(metrics_file_path, "w+") as handle:
        print("Epoch\tTrain loss\tTrain acc\tDev loss\tDev acc", file=handle)

    print("Loading the training dataset")
    train_data = load_and_cache_data(processor, train_file, cache_path)
    train_dataset = Seq2SeqDataset(train_data)

    print("Loading the validation dataset")
    valid_data = load_and_cache_data(processor, valid_file, cache_path)
    valid_dataset = Seq2SeqDataset(valid_data)

    optimizer = torch.optim.Adam(model.parameters())
    best_valid_loss = float("inf")
    epoch_iterator = range(1, epochs + 1)
    for epoch in epoch_iterator:
        print("¤ Epoch %d / %d" % (epoch, epochs))

        train_it = Seq2SeqDataLoader(
            train_dataset,
            pad=processor.pad_token_id,
            batch_size=batch_size,
            shuffle=True,
            device=device,
        )
        train_it = tqdm(train_it, desc="├ Train", file=sys.stdout)
        train_loss, train_met = model.run_epoch(
            train_it, optimizer, teacher_forcing_ratio=0.5
        )
        print("│ └ Loss: %.3f | %s" % (train_loss, metrics_str(train_met)))

        valid_it = Seq2SeqDataLoader(
            valid_dataset,
            pad=processor.pad_token_id,
            batch_size=batch_size,
            device=device,
        )
        valid_it = tqdm(valid_it, desc="└ Valid", file=sys.stdout)
        valid_loss, valid_met = model.run_epoch(valid_it)
        if valid_loss >= best_valid_loss or math.isinf(best_valid_loss):
            print("  └ Loss: %.3f | %s" % (valid_loss, metrics_str(valid_met)))
        else:
            delta_loss = valid_loss - best_valid_loss
            print(
                "  └ Loss: %.3f (%.3fΔ) | %s"
                % (valid_loss, delta_loss, metrics_str(valid_met))
            )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_file_path)

        with open(metrics_file_path, "a") as handle:
            print(
                "%d\t%.4f\t%.4f" % (epoch, train_loss, valid_loss), file=handle,
            )


def test(
    run_id: str,
    batch_size: int,
    cache_path: str,
    output_path: str,
    test_file: str,
):
    print("Starting testing with run id %s" % run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_path, exist_ok=True)

    run_path = os.path.join(output_path, run_id)
    config_file_path = os.path.join(run_path, "config.pkl")
    model_file_path = os.path.join(run_path, "model.pt")

    config = Seq2SeqConfig.load(config_file_path)
    processor = Seq2SeqProcessor.from_config(config.processor)
    model = Seq2SeqModel.from_config(config.model).to(device)
    model.load_state_dict(torch.load(model_file_path, map_location=device))

    print("Loading the test dataset")
    test_data = load_and_cache_data(processor, test_file, cache_path)
    test_dataset = Seq2SeqDataset(test_data)

    print("¤ Evaluation ")
    test_it = Seq2SeqDataLoader(
        test_dataset,
        pad=processor.pad_token_id,
        batch_size=batch_size,
        device=device,
    )
    test_it = tqdm(test_it, desc="├ Test", file=sys.stdout)

    test_loss, test_met = model.run_epoch(test_it)
    print("  └ Loss: %.3f | %s" % (test_loss, metrics_str(test_met)))


def main(args: Dict[str, Any]) -> None:
    set_seed(args.seed)

    os.makedirs(args.cache_path, exist_ok=True)

    if args.do_train:
        train(
            args.run_id,
            args.epochs,
            args.batch_size,
            args.cache_path,
            args.output_path,
            args.train_file,
            args.valid_file,
        )

    if args.do_test:
        test(
            args.run_id,
            args.batch_size,
            args.cache_path,
            args.output_path,
            args.test_file,
        )


if __name__ == "__main__":
    try:
        args = parse_args()

        normalize_args(args)
        validate_args(args)
        main(args)
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted!")
