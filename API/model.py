import torch
import torch.nn as nn


class PriceLSTM(nn.Module):
    def __init__(self, features_count, output_size=1, sequence_size=10, n_layers=1, drop_prob=0.5,
                 device=torch.device("cpu")):
        super(PriceLSTM, self).__init__()

        embedding_dim = features_count * 2
        hidden_dim = features_count

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.liniar = nn.Linear(features_count, embedding_dim)
        self.normalization_pre_lstm = nn.BatchNorm1d(num_features=sequence_size)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                nn.BatchNorm1d(num_features=hidden_dim // 2),
                                nn.LeakyReLU(),
                                nn.Dropout(drop_prob),
                                nn.Linear(hidden_dim // 2, output_size))

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        out = self.liniar(x)
        out = self.normalization_pre_lstm(out)
        out = torch.tanh(out)
        out, (hn, cn) = self.lstm(out, (h0, c0))
        out = torch.add(out, x)
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])

        return out

    @classmethod
    def default(cls, features_count, device, seq_size):
        return cls(features_count=features_count,
                   output_size=1, # for each sequence, one output
                   device=device, n_layers=4, sequence_size=seq_size, drop_prob=0.4)
