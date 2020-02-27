from functools import partial
from typing import *  # pylint: disable=W0401,W0614

import torch


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # source_seq, target_seq
        example = self.data[index]
        return example[0], example[1]


class Seq2SeqDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kargs: Dict[str, Any]) -> None:
        kargs["collate_fn"] = partial(
            self.pad_collate_fn,
            pad=kargs.pop("pad", 0),
            batch_first=kargs.pop("batch_first", False),
            sort_key=kargs.pop("sort_key", lambda x: len(x[0])),
            device=kargs.pop("device", None),
        )
        super(Seq2SeqDataLoader, self).__init__(dataset, **kargs)

    @classmethod
    def pad_collate_fn(
        cls,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
        pad: int = 0,
        batch_first: bool = False,
        sort_key: Optional[
            Callable[[Tuple[torch.Tensor, torch.Tensor]], Any]
        ] = None,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[int]]:
        """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
        We build a custom collate_fn rather than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Seqeuences are padded to the length provided or to the maximum length
        of mini-batch sequences (dynamic padding).
        Args:
            batch: list of tuple (src_seq, trg_seq).
                - src_seq: torch tensor of shape (?); variable length.
                - trg_seq: torch tensor of shape (?); variable length.
            pad: value to use as padding.
            batch_first: if True, then the input and output tensors are provided
            as (batch_size, padded_length).
            sort_key: A key to use for sorting examples in order to batch
            together examples with similar lengths and minimize padding.
        Returns:
            src_seqs: torch tensor of shape (padded_length, batch_size).
            src_lengths: list of length (batch_size); valid length for each
            padded source sequence.
            trg_seqs: torch tensor of shape (padded_length, batch_size).
            trg_lengths: list of length (batch_size); valid length for each
            padded target sequence.
        """

        def merge(sequences, batch_first, pad, device):
            lengths = [len(seq) for seq in sequences]
            batch_size = len(sequences)
            padded_length = max(lengths)
            if batch_first:
                padded_seqs = torch.full(
                    (batch_size, padded_length),
                    pad,
                    dtype=torch.long,
                    device=device,
                )
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            else:
                padded_seqs = torch.full(
                    (padded_length, batch_size),
                    pad,
                    dtype=torch.long,
                    device=device,
                )
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[:end, i] = seq[:end]
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            return (padded_seqs, lengths)

        # sort a list by sequence length (descending order)
        # to use pack_padded_sequence
        if sort_key is not None:
            batch.sort(key=sort_key, reverse=True)

        # seperate source and target sequences
        src_seqs, trg_seqs = zip(*batch)

        # merge sequences
        src_seqs, src_lengths = merge(src_seqs, batch_first, pad, device)
        trg_seqs, trg_lengths = merge(trg_seqs, batch_first, pad, device)

        return src_seqs, src_lengths, trg_seqs, trg_lengths
