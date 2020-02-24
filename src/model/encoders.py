from typing import *

import torch


class RNNEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        rnn_cell="lstm",
        embedding_dropout=0.0,
        layers_dropout=0.0,
    ):
        super(RNNEncoder, self).__init__()

        if rnn_cell.lower() == "lstm":
            self.rnn_cell = torch.nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = torch.nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, num_layers, dropout=layers_dropout
        )
        self.dropout = torch.nn.Dropout(embedding_dropout)

    # input = [seq_len, batch_size]
    # input_len = [batch_size]
    # hidden = (h_n, c_n) if LSTM or h_n ortherwise
    # h_n = [num_layers, batch_size, hidden_dim]
    # c_n = [num_layers, batch_size, hidden_dim]
    def forward(self, input, input_len, hidden=None):
        # embedded = [seq_len, batch_size, embbedding_dim]
        embedded = self.dropout(self.embedding(input))

        # outputs = [seq_len, batch_size, hidden_dim]
        # hidden = (h_n, c_n) if LSTM or h_n ortherwise
        # h_n = [num_layers, batch_size, hidden_dim]
        # c_n = [num_layers, batch_size, hidden_dim]
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_len
        )
        packed_outputs, hidden = self.rnn(packed_embedded, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs)
        return outputs, hidden

    def name(self) -> str:
        cell = "lstm" if self.rnn_cell is torch.nn.LSTM else "gru"
        return cell + "(%d,%d,%d,%d)" % (
            self.input_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
        )