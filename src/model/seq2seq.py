import random
from contextlib import nullcontext
from collections import defaultdict
from typing import *

import editdistance
import torch

from model.config import Seq2SeqConfig
from model.encoders import RNNEncoder
from model.decoders import RNNDecoder


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
        target_seq: Optional[torch.Tensor] = None,
        target_length: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: Optional[float] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = source_seq.device
        batch_size = source_seq.shape[1]
        target_len = 1 if target_seq is None else target_seq.shape[0]
        output_size = self.decoder.output_dim
        max_len = self.output_seq_max_length
        trf = teacher_forcing_ratio

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, output_size, device=device)
        outputs[0, :, self.bos_token_id] = torch.tensor(1.0)
        outputs[1:, :, self.eos_token_id] = torch.tensor(1.0)

        preds = torch.full(
            (max_len, batch_size),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        preds[0, :] = self.bos_token_id

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        _, hidden = self.encoder(source_seq, source_length)

        # first input to the decoder is the BOS tokens
        input = target_seq[0, :]

        ended = None
        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell
            # states receive output tensor (predictions) and new hidden and cell
            # states
            output, hidden = self.decoder(input.detach(), hidden)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = (
                t < target_len and trf > 0.0 and random.random() < trf
            )

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            pred = output.argmax(dim=-1)
            preds[t] = pred.detach()
            input = target_seq[t] if teacher_force else pred

            if t >= target_len:
                if ended is None:
                    ended = preds.eq(self.eos_token_id).sum(0) > 0
                else:
                    ended = ended | preds[t].eq(self.eos_token_id)
                if ended.sum().item() == batch_size:
                    break

        return preds, outputs

    def predict(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs.argmax(dim=-1)

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
        metrics = defaultdict(lambda: (0, 0))
        with nullcontext() if optimizer is not None else torch.no_grad():
            for batch in iterator:
                src, src_len, trg, trg_len = batch

                if optimizer is not None:
                    # Zero gradients
                    optimizer.zero_grad()

                # Forward pass through the seq2seq model
                tfr = teacher_forcing_ratio
                pred, outputs = self.forward(src, src_len, trg, trg_len, tfr)

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
                    pred_mask = (pred == self.eos_token_id).cumsum(dim=0) <= 1
                    pred_len = pred_mask.sum(dim=0)
                    trg_mask = trg.ne(self.pad_token_id)
                    new_metrics = self.__compute_metrics(
                        pred, pred_len, pred_mask, trg, trg_len, trg_mask
                    )
                    for key in new_metrics:
                        metrics[key] = tuple(
                            map(sum, zip(metrics[key], new_metrics[key]))
                        )

        total_loss = total_loss[0] / total_loss[1]
        for key in metrics:
            metrics[key] = metrics[key][0] / metrics[key][1] * 100

        return total_loss, metrics

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

    def __compute_metrics(
        self,
        pred: torch.Tensor,
        pred_len: torch.Tensor,
        pred_mask: torch.Tensor,
        trg: torch.Tensor,
        trg_len: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> Dict[str, Tuple[int, int]]:
        trg_max_len = trg.shape[0]
        batch_size = pred.shape[1]
        combined_mask = pred_mask[:trg_max_len] & trg_mask

        eq = pred[:trg_max_len].eq(trg)

        # Teacher Accuracy
        tacc_correct = eq.masked_select(trg_mask).sum().item()
        tacc_total = trg_len.sum().item()

        # Prediction Accuracy
        acc_correct = eq.masked_select(combined_mask).sum().item()
        acc_total = torch.max(pred_len, trg_len).sum().item()

        # Edit distance
        edist_correct, edist_total = acc_total, acc_total
        for b in range(0, batch_size):
            prediction, target = (pred[: pred_len[b], b], trg[: trg_len[b], b])
            edist = editdistance.eval(prediction, target)
            edist_correct -= edist

        return {
            "tacc": (tacc_correct, tacc_total),
            "acc": (acc_correct, acc_total),
            "edist": (edist_correct, edist_total),
        }

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
