import random
from contextlib import nullcontext
from typing import *

import editdistance
import torch

from model.config import Seq2SeqConfig
from model.encoders import RNNEncoder
from model.decoders import RNNDecoder

from utils.torch import find_first


class Seq2SeqModel(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        mask_token_id: int,
        input_seq_max_length: int,
        output_seq_max_length: int,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.input_seq_max_length = input_seq_max_length
        self.output_seq_max_length = output_seq_max_length

    # source_seq = [source_seq_len, batch_size]
    # target_seq = [target_seq_len, batch_size]
    def forward(
        self,
        source_seq: torch.Tensor,
        source_length: torch.Tensor,
        target_seq: torch.Tensor,
        target_length: torch.Tensor,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        device = target_seq.device
        batch_size = target_seq.shape[1]
        target_len = target_seq.shape[0]
        output_size = self.decoder.output_dim
        max_len = self.output_seq_max_length
        trf = teacher_forcing_ratio

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, output_size, device=device)
        outputs[0, :, target_seq[0, 0]] = torch.tensor(1.0)

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        _, hidden = self.encoder(source_seq, source_length)

        # first input to the decoder is the BOS tokens
        input = target_seq[0, :]

        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell
            # states receive output tensor (predictions) and new hidden and cell
            # states
            output, hidden = self.decoder(input.detach(), hidden)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = t < target_len and random.random() < trf

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = target_seq[t] if teacher_force else output.argmax(dim=-1)

        return outputs

    def predict(
        self, outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = outputs.argmax(dim=-1)
        _, first_eos = find_first(pred, self.eos_token_id, axis=0)
        pred_len = first_eos + 1
        return pred, pred_len

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def name(self) -> str:
        enc = self.encoder.name()
        dec = self.decoder.name()
        return "seq2seq(%s,%s)" % (enc, dec)

    def run_epoch(
        self,
        iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        teacher_forcing_ratio: float = 0.0,
        clip: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        assert teacher_forcing_ratio == 0.0 or optimizer is not None
        assert 0.0 <= teacher_forcing_ratio and teacher_forcing_ratio <= 1.0

        self.train() if optimizer is not None else self.eval()

        total_loss = (0, 0)
        metrics = {"tacc": (0, 0), "acc": (0, 0), "edist": (0, 0)}
        with nullcontext() if optimizer is not None else torch.no_grad():
            for batch in iterator:
                src, src_len, trg, trg_len = batch

                if optimizer is not None:
                    # Zero gradients
                    optimizer.zero_grad()

                # Forward pass through the seq2seq model
                tfr = teacher_forcing_ratio
                outputs = self.forward(src, src_len, trg, trg_len, tfr)

                # Calculate and accumulate the loss
                loss_outputs = outputs[: trg.shape[0], :, :]
                loss = self.criterion(
                    loss_outputs.view(-1, loss_outputs.shape[-1]), trg.view(-1)
                )
                total_loss = tuple(map(sum, zip(total_loss, (loss.item(), 1))))

                if optimizer is not None:
                    # Perform backpropatation, clip gradients, and adjust model weights
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                    optimizer.step()

                with nullcontext() if optimizer is None else torch.no_grad():
                    # Compute additional metrics
                    pred, pred_len = self.predict(outputs)
                    tacc = self.__compute_tacc(pred, pred_len, trg, trg_len)
                    acc = self.__compute_acc(pred, pred_len, trg, trg_len)
                    edist = self.__compute_edist(pred, pred_len, trg, trg_len)
                    metrics = {
                        "tacc": tuple(map(sum, zip(metrics["tacc"], tacc))),
                        "acc": tuple(map(sum, zip(metrics["acc"], acc))),
                        "edist": tuple(map(sum, zip(metrics["edist"], edist))),
                    }

        for key in metrics:
            metrics[key] = metrics[key][0] / metrics[key][1] * 100

        return total_loss[0] / total_loss[1], metrics

    def run_prediction(
        self, sequence: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        device = sequence.device

        self.eval()
        with torch.no_grad():
            src = sequence.unsqueeze(1)
            src_len = torch.tensor([len(sequence)], device=device)
            preds = torch.full(
                (max_len,), self.pad_token_id, dtype=torch.long, device=device
            )

            _, hidden = self.encoder(src, src_len)
            input = torch.tensor([self.bos_token_id], device=device)
            preds[0] = input[0].item()
            for t in range(1, max_len):
                output, hidden = self.decoder(input, hidden)
                pred = output.argmax(dim=-1)
                preds[t] = pred.item()
                if preds[t] == self.eos_token_id:
                    return preds[: t + 1]
                input = pred
        return preds

    def __compute_tacc(
        self,
        pred: torch.Tensor,
        pred_len: torch.Tensor,
        trg: torch.Tensor,
        trg_len: torch.Tensor,
    ) -> Tuple[int, int]:
        trg_max_len = trg.shape[0]
        no_pad = trg.ne(self.pad_token_id)
        correct = (
            pred[:trg_max_len, :].eq(trg).masked_select(no_pad).sum().item()
        )
        total = no_pad.sum().item()
        return correct, total

    def __compute_acc(
        self,
        pred: torch.Tensor,
        pred_len: torch.Tensor,
        trg: torch.Tensor,
        trg_len: torch.Tensor,
    ) -> Tuple[int, int]:
        batch_size = pred.shape[1]

        correct, total = 0, 0
        for b in range(0, batch_size):
            min_len = min(pred_len[b].item(), trg_len[b].item())
            max_len = max(pred_len[b].item(), trg_len[b].item())
            correct += pred[:min_len, b].eq(trg[:min_len, b]).sum().item()
            total += max_len
        return correct, total

    def __compute_edist(
        self,
        pred: torch.Tensor,
        pred_len: torch.Tensor,
        trg: torch.Tensor,
        trg_len: torch.Tensor,
    ) -> Tuple[int, int]:
        batch_size = pred.shape[1]

        correct, total = pred_len.sum().item(), trg_len.sum().item()
        for b in range(0, batch_size):
            prediction, target = pred[: pred_len[b], b], trg[: trg_len[b], b]
            edist = editdistance.eval(prediction, target)
            correct -= edist
        return correct, total

    @classmethod
    def from_config(cls, config: Seq2SeqConfig):
        if config.encoder.name == "rnn":
            enc = RNNEncoder(
                config.encoder.vocab_size,
                config.encoder.embedding_size,
                config.encoder.hidden_size,
                num_layers=config.encoder.num_layers,
                layers_dropout=config.encoder.layers_dropout,
                embedding_dropout=config.encoder.embedding_dropout,
                rnn_cell=config.encoder.rnn_cell,
            )
        else:
            raise NotImplementedError()
        if config.decoder.name in "rnn":
            dec = RNNDecoder(
                config.decoder.vocab_size,
                config.decoder.embedding_size,
                config.decoder.hidden_size,
                num_layers=config.decoder.num_layers,
                layers_dropout=config.decoder.layers_dropout,
                embedding_dropout=config.decoder.embedding_dropout,
                rnn_cell=config.decoder.rnn_cell,
            )
        else:
            raise NotImplementedError()

        special_tokens = {
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
            "mask_token_id": config.mask_token_id,
            "input_seq_max_length": config.input_seq_max_length,
            "output_seq_max_length": config.output_seq_max_length,
        }

        return cls(enc, dec, **special_tokens)
