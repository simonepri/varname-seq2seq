import random
from contextlib import nullcontext
from typing import *

import torch

from utils.configs import Config
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

        assert encoder.hidden_dim == decoder.hidden_dim
        assert encoder.num_layers == decoder.num_layers
        assert encoder.rnn_cell == decoder.rnn_cell

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
        batch_size = target_seq.shape[1]
        target_len = target_seq.shape[0]
        output_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(
            target_len, batch_size, output_size, device=target_seq.device
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
        device: torch.device = None,
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

    @classmethod
    def from_config(cls, config: Config):
        if config.encoder_name in ["gru", "lstm"]:
            enc = RNNEncoder(
                config.vocab_size,
                config.embedding_size,
                config.hidden_size,
                num_layers=config.num_layers,
                layers_dropout=config.layers_dropout,
                embedding_dropout=config.embedding_dropout,
                rnn_cell=config.encoder_name,
            )
        else:
            raise NotImplementedError()
        if config.decoder_name in ["gru", "lstm"]:
            dec = RNNDecoder(
                config.vocab_size,
                config.embedding_size,
                config.hidden_size,
                num_layers=config.num_layers,
                layers_dropout=config.layers_dropout,
                embedding_dropout=config.embedding_dropout,
                rnn_cell=config.decoder_name,
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
