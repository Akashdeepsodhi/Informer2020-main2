import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FeatureExtractor(nn.Module):
    def __init__(self, c_in, m=7, tau=3):
        super(FeatureExtractor, self).__init__()
        self.m = m
        self.tau = tau
        self.window_size = m + 1  # Current time step + m * tau
        self.c_in = c_in

    def forward(self, x):
        batch_size, seq_length, c_in = x.shape
        assert c_in == self.c_in, "Input channels do not match initialization."

        valid_t = seq_length - (self.m * self.tau)
        feature_vectors = []

        for t in range(valid_t):
            indices = [t + i * self.tau for i in range(self.m + 1)]
            window = x[:, indices, :]  # Shape: (batch_size, m+1, c_in)
            flat_window = window.reshape(batch_size, -1)  # Shape: (batch_size, c_in * (m+1))
            feature_vectors.append(flat_window)

        output = torch.stack(feature_vectors, dim=1)  # Shape: (batch_size, valid_t, c_in * (m+1))
        return output


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=7, tau=3):
        super(TokenEmbedding, self).__init__()
        self.feature_extractor = FeatureExtractor(c_in, m=m, tau=tau)
        self.linear = nn.Linear(c_in * (m + 1), d_model)

    def forward(self, x):
        features = self.feature_extractor(x)  # Shape: (batch_size, valid_t, c_in * (m+1))
        embeddings = self.linear(features)  # Shape: (batch_size, valid_t, d_model)
        return embeddings


class CircularConv1D(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super(CircularConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )

    def circular_pad(self, x):
        padding_size = self.kernel_size - 1
        x = torch.cat([x[:, :, -padding_size:], x, x[:, :, :padding_size]], dim=2)
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Shape: (batch_size, c_in, seq_length)
        x = self.circular_pad(x)
        x = self.conv(x)
        return x.permute(0, 2, 1)  # Shape: (batch_size, seq_length, d_model)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Step 1: Check and adjust sequence length (seq_len)
        seq_len = x.size(1)
        if seq_len < 96:
            # Pad the sequence to 96
            pad_size = 96 - seq_len
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))  # Pad along the time dimension
            x_mark = F.pad(x_mark, (0, 0, 0, 0, 0, pad_size))

        elif seq_len > 96:
            # Truncate the sequence to 96
            x = x[:, :96, :]
            x_mark = x_mark[:, :96, :]

        # Step 2: Apply embeddings
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)


