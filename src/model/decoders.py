from typing import *

import torch


class RNNDecoder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        rnn_cell="lstm",
        embedding_dropout=0.0,
        layers_dropout=0.0,
    ):
        super(RNNDecoder, self).__init__()

        if rnn_cell.lower() == "lstm":
            self.rnn_cell = torch.nn.LSTM
        elif rnn_cell.lower() == "gru":
            self.rnn_cell = torch.nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(output_dim, embedding_dim)
        self.rnn = self.rnn_cell(
            embedding_dim, hidden_dim, num_layers, dropout=layers_dropout
        )
        self.out = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(embedding_dropout)

    # input = [batch_size]
    # hidden = (h_n, c_n) if LSTM or h_n ortherwise
    # h_n = [num_layers, batch_size, hidden_dim]
    # c_n = [num_layers, batch_size, hidden_dim]
    def forward(self, input, input_len, hidden=None):
        # input = [1, batch_size]
        input = input.unsqueeze(0)

        # embedded = [1, batch_size, embedding_dim]
        embedded = self.dropout(self.embedding(input))

        # output = [1, batch_size, hidden_dim]
        # hidden = (h_n, c_n) if LSTM or h_n ortherwise
        # h_n = [num_layers, batch_size, hidden_dim]
        # c_n = [num_layers, batch_size, hidden_dim]
        output, hidden = self.rnn(embedded, hidden)

        # prediction = [batch_size, output dim]
        prediction = self.out(output.squeeze(0))

        return prediction, hidden

    def name(self) -> str:
        cell = "lstm" if self.rnn_cell is torch.nn.LSTM else "gru"
        return cell + "(%d,%d,%d,%d)" % (
            self.output_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
        )
