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

    def name(self) -> str:
        """The name of the model given the current cofiguration.

        Returns:
            (str): a string containing the name of the model.

        """
        enc = self.encoder.name()
        dec = self.decoder.name()
        return "seq2seq(%s,%s)" % (enc, dec)

    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        trg: Optional[torch.Tensor] = None,
        trg_len: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: Optional[float] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward.

        Args:
            src (torch.long): Tensor of shape [SL, B], representing the source
                token sequences.
            src_len (torch.long): Tensor of shape [B], representing the
                non-padded length of each source sequence.
            trg (torch.long, optional): Tensor of shape [TL, B], representing
                the target token sequences.
            trg_len (torch.long, optional): Tensor of shape [B], representing
                the non-padded length of each target sequence.
            teacher_forcing_ratio (float, optional): The percentage of times we
                use teacher forcing during decoding.

        Returns:
            (tuple): tuple containing:
                predictions (torch.long): Tensor of shape [?, B], containing the
                    predicted token sequences.

                outputs (torch.float): Tensor of shape [?, B, V], storing the
                    distribution over output symbols for each timestep for each
                    batch element.

        Note:
            In the source and target token sequences we assume that the first
            token is the BOS token and that they always contain the EOS token
            before the padding.
            The first predicted token is always the BOS token.
            Insted, the EOS token might not apper in the predictions, meaning
            that we truncated the decoder output to the maximum output length.
            When the target sequence is not provided we assume we are in
            prediction mode, so we keep forwarding inputs to the decoder as long
            as can.

        """
        device = src.device

        teach_len = 1 if trg is None else trg.shape[0]
        batch_size = src.shape[1]
        output_size = self.decoder.output_dim
        max_len = self.output_seq_max_length
        trf = teacher_forcing_ratio

        # Last hidden state of the encoder is used as the initial hidden state
        # of the decoder.
        _, hidden = self.encoder(src, src_len)

        # The first input of the decoder is the BOS tokens.
        input = torch.full(
            (batch_size,), self.bos_token_id, dtype=torch.long, device=device,
        )
        # Tensor to store decoder outputs, the first output is BOS.
        outputs = torch.empty(teach_len, batch_size, output_size, device=device)
        outputs[0, :, :] = 0.0
        outputs[0, :, self.bos_token_id] = 1.0
        # Tensor to store decoder predictions, the first prediction is BOS.
        predictions = torch.empty(
            teach_len, batch_size, dtype=torch.long, device=device,
        )
        predictions[0, :] = self.bos_token_id

        # If a target sequence is provided, predict at least teach_len tokens.
        # At each step, we alternate the use of the actual next token as next
        # input with the predicted token from the previous step.
        for t in range(1, teach_len):
            output, hidden = self.decoder(input.detach(), hidden)
            outputs[t] = output
            teacher_force = trf > 0.0 and random.random() < trf
            pred = output.argmax(dim=-1)
            predictions[t] = pred
            input = trg[t] if teacher_force else pred

        # Keep predicting until all the sequences contain and EOS token or we
        # predicted max_len tokens for all the sequences.
        ended = predictions.eq(self.eos_token_id).sum(dim=0) > 0
        for t in range(teach_len, max_len):
            if ended.sum().item() == batch_size:
                break
            output, hidden = self.decoder(input.detach(), hidden)
            outputs = torch.cat((outputs, output.unsqueeze(dim=0)), dim=0)
            pred = output.argmax(dim=-1)
            predictions = torch.cat((predictions, pred.unsqueeze(dim=0)), dim=0)
            input = pred
            ended = ended | pred.eq(self.eos_token_id)

        return predictions, outputs

    def run_epoch(
        self,
        iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        teacher_forcing_ratio: Optional[float] = 0.0,
        clip: Optional[float] = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Run a single epoch.

        Args:
            iterator (iterable): iterator generating:
                batch (tuple): tuple containing:
                    src (torch.long): Tensor of shape [SL, B], representing the
                        source token sequences.
                    src_len (torch.long): Tensor of shape [B], representing
                        the non-padded length of each source sequence.
                    trg (torch.long): Tensor of shape [TL, B],
                        representing the target token sequences.
                    trg_len (torch.long): Tensor of shape [B],
                        representing the non-padded length of each target
                        sequence.
            optimizer (torch.optim.Optimizer): A pytorch optimizer. If not
                provided we assume we running the model in evaluation mode.
            teacher_forcing_ratio (float, optional): The percentage of times we
                use teacher forcing during decoding. This should not be provided
                if the optimizer is not provided as well.
            clip (float, optional): When the optimizer is provided, this value
                is used to clip the gradients to avoid exploding gradients.

        Returns:
            (tuple): tuple containing:
                epoch_loss (float): The average per batch loss of the current
                    epoch. The value returned during training may differ from
                    the value in evaluation mode, because we do backpropatation
                    after every batch.
                epoch_metrics (dict): A dictionary containing different metrics
                    for the current epoch. This dictionary is only populated in
                    evaluation mode (i.e. when an optimizer is not provided).
                    Metrics returned include:
                        "tacc": target accuracy, i.e. simple accuracy using
                        the target sequence.
                        "acc": similar to "tacc", but it also counts as errors
                        tokens predicted after the target sequence (i.e. it
                        penalizes the model when it does not predict the EOS
                        token correctly).
                        "edist": percentage of correct token given the
                        Levenshtein distance between the target and predicted
                        sequence.
                    Note that all the metrics include the EOS token in the
                    computation, so neither of them can never be zero.

        Note:
            In the source and target token sequences we assume that the first
            token is the BOS token and that they always contain the EOS token
            before the padding.
            The tensor inside the batch are expected to be already on the
            correct device (i.e. cuda or cpu).

        """
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
                pred, out = self.forward(src, src_len, trg, trg_len, tfr)

                # Calculate and accumulate the loss
                loss_out = out[1 : trg.shape[0], :, :]
                loss = self.criterion(
                    loss_out.view(-1, loss_out.shape[-1]), trg[1:].view(-1),
                )
                total_loss = tuple(map(sum, zip(total_loss, (loss.item(), 1))))

                if optimizer is not None:
                    # Perform backpropatation, clip gradients, and adjust model
                    # weights
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                    optimizer.step()
                else:
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

        return total_loss, dict(metrics.items())

    def run_prediction(self, src: torch.Tensor, src_len: int) -> torch.Tensor:
        """Generate predictions.

        Args:
            src (torch.long): Tensor of shape [SL], representing the source
                token sequence.
            src_len (int): the non-padded length of the source sequence.

        Returns:
            predictions (torch.long): Tensor of shape [?], containing the
            predicted token sequence.

        Note:
            The source tensor is expected to be already on the correct device
            (i.e. cuda or cpu).

        """
        self.eval()
        with torch.no_grad():
            src = src.unsqueeze(dim=-1)
            src_len = torch.full(
                (1,), src_len, dtype=torch.long, device=src.device
            )
            pred, _ = self.forward(src, src_len)

        return pred

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
