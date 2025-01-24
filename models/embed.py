import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
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
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, c_in)

        Returns:
            output: Tensor of shape (batch_size, valid_t, c_in * (m + 1))
        """
        batch_size, seq_length, c_in = x.shape
        assert c_in == self.c_in, "Input channels do not match initialization."

        # Define the valid length of time steps after applying the window
        valid_t = seq_length - (self.m * self.tau)
        feature_vectors = []

        # Extract features for each t
        for t in range(valid_t):
            # Slice the input tensor for the window [t, t + m*tau]
            indices = [t + i * self.tau for i in range(self.m + 1)]
            window = x[:, indices, :]  # Shape: (batch_size, m+1, c_in)

            # Flatten the window for each time step
            flat_window = window.reshape(batch_size, -1)  # Shape: (batch_size, c_in * (m+1))
            feature_vectors.append(flat_window)

        # Stack all feature vectors along the time dimension
        output = torch.stack(feature_vectors, dim=1)  # Shape: (batch_size, valid_t, c_in * (m+1))
        return output

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, m=7, tau=3):
        """
        Args:
            c_in: Number of input features/channels.
            d_model: Dimension of the final token embeddings.
            m: Number of future steps to consider.
            tau: Stride for future steps.
        """
        super(TokenEmbedding, self).__init__()
        self.feature_extractor = FeatureExtractor(c_in, m=m, tau=tau)
        self.linear = nn.Linear(c_in * (m + 1), d_model)  # Project to desired embedding size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, c_in)
        
        Returns:
            embeddings: Tensor of shape (batch_size, valid_t, d_model)
        """
        # Extract contextual features
        features = self.feature_extractor(x)  # Shape: (batch_size, valid_t, c_in * (m+1))

        # Project to desired embedding dimension
        embeddings = self.linear(features)  # Shape: (batch_size, valid_t, d_model)
        return embeddings

class CircularConv1D(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        """
        Args:
            c_in: Number of input channels.
            d_model: Desired number of output features (output channels).
            kernel_size: Size of the convolutional kernel.
        """
        super(CircularConv1D, self).__init__()
        self.kernel_size = kernel_size

        # Define the 1D convolution layer
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,  # Default stride is 1
            padding=0  # Padding will be added manually
        )

    def circular_pad(self, x):
        """
        Manually apply circular padding.
        Args:
            x: Input tensor of shape (batch_size, c_in, seq_length)
        Returns:
            Padded tensor of shape (batch_size, c_in, seq_length + 2 * (kernel_size - 1))
        """
        padding_size = self.kernel_size - 1
        # Concatenate slices from the end and start for circular padding
        x = torch.cat([x[:, :, -padding_size:], x, x[:, :, :padding_size]], dim=2)
        return x

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, c_in)
        Returns:
            output: Tensor of shape (batch_size, d_model, valid_t)
        """
        # Reshape to (batch_size, c_in, seq_length) for Conv1d
        x = x.permute(0, 2, 1)

        # Apply circular padding
        x = self.circular_pad(x)

        # Perform convolution
        x = self.conv(x)

        # Compute valid_t
        valid_t = x.shape[2] - (self.kernel_size - 1)

        # Retain only valid_t time steps
        return x[:, :, :valid_t]


class EmbeddingPipeline(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3, m=7, tau=3):
        """
        Args:
            c_in: Number of input channels/features.
            d_model: Dimension of the final token embeddings.
            kernel_size: Size of the convolution kernel.
            m: Number of future steps to consider for feature extraction.
            tau: Stride for future steps.
        """
        super(EmbeddingPipeline, self).__init__()
        self.feature_extractor = FeatureExtractor(c_in, m, tau)
        self.circular_conv = CircularConv1D(c_in * (m + 1), d_model, kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, c_in)
        Returns:
            embeddings: Tensor of shape (batch_size, d_model, valid_t)
        """
        # Step 1: Extract contextual feature vectors
        features = self.feature_extractor(x)  # Shape: (batch_size, valid_t, c_in * (m+1))

        # Step 2: Perform circular convolution to produce embeddings
        embeddings = self.circular_conv(features)  # Shape: (batch_size, d_model, valid_t)
        return embeddings

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
