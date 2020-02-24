import random
from contextlib import nullcontext
from typing import *

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
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    # source_seq = [source_seq_len, batch_size]
    # target_seq = [target_seq_len, batch_size]
    def forward(
        self,
        source_seq: torch.Tensor,
        source_length: torch.Tensor,
        target_seq: torch.Tensor,
        target_length: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        device = target_seq.device
        batch_size = target_seq.shape[1]
        target_len = target_seq.shape[0]
        output_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(
            target_len, batch_size, output_size, device=device
        )
        outputs[0, :, target_seq[0, 0]] = torch.tensor(1.0)

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        _, hidden = self.encoder(source_seq, source_length)

        # first input to the decoder is the BOS tokens
        input = target_seq[0, :]

        for t in range(1, target_len):
            # insert input token embedding, previous hidden and previous cell
            # states receive output tensor (predictions) and new hidden and cell
            # states
            output, hidden = self.decoder(input, hidden)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = target_seq[t] if teacher_force else output.argmax(dim=-1)

        return outputs

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
        clip: float = 1.0,
    ) -> Tuple[float, float]:
        self.train() if optimizer is not None else self.eval()

        epoch_correct_preds = 0
        epoch_total_preds = 0
        epoch_loss = 0.0
        with nullcontext() if optimizer is not None else torch.no_grad():
            for batch in iterator:
                src, src_len, trg, trg_len = batch

                if optimizer is not None:
                    # Zero gradients
                    optimizer.zero_grad()

                # Forward pass through the seq2seq model
                outputs = self.forward(src, src_len, trg, trg_len)

                # Calculate and accumulate the loss
                loss = self.criterion(
                    outputs.view(-1, outputs.shape[-1]), trg.view(-1)
                )
                epoch_loss += loss.item()

                # Count number of correct tokens
                preds = outputs.argmax(dim=-1)
                no_pad = trg.ne(self.pad_token_id)
                epoch_correct_preds += (
                    preds.eq(trg).masked_select(no_pad).sum().item()
                )
                epoch_total_preds += trg_len.sum().item()

                if optimizer is not None:
                    # Perform backpropatation
                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

                    # Adjust model weights
                    optimizer.step()

        final_loss = epoch_loss / len(iterator)
        final_acc = epoch_correct_preds / epoch_total_preds
        return final_loss, final_acc

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
        }

        return cls(enc, dec, **special_tokens)
