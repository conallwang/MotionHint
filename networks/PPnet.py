import torch
from torch import nn


class PPnet(nn.Module):
    def __init__(self, input_size=6, output_size=6, seq=20, hidden_size=8, num_layer=1, batch_first=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq = seq

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=batch_first)

        self.fn_mean = nn.Linear(seq * hidden_size, output_size)
        self.fn_variance = nn.Linear(seq * hidden_size, output_size)

    def forward(self, x):
        t_feature, (_, _) = self.lstm(x)

        t_feature = t_feature.reshape(-1, self.hidden_size * self.seq)

        mean = self.fn_mean(t_feature)             # [BS, 6]
        variance = self.fn_variance(t_feature)     # [BS, 6]

        return mean, variance
