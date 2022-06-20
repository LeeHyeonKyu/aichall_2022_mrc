import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class QAConvSDSLayer(nn.Module):
    """Conv SDS layer for qa output"""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
    ):
        """
        Args:
            input_size (int): max sequence lengths
            hidden_dim (int): backbones's hidden dimension
        """

        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size * 2,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=input_size * 2,
            out_channels=input_size,
            kernel_size=1,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + torch.relu(out)
        out = self.layer_norm(out)
        return out


class AttentionLayer(nn.Module):
    """Attention for query embedding"""

    def __init__(self, config, need_query_emb=False):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__()
        self.query_rnn = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.query_layer = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=True
        )
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.need_query_emb = need_query_emb

    def forward(self, x: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Layer input
            token_type_ids (torch.Tensor): Token type ids of input_ids
        Returns:
            torch.Tensor: embedded value (batch_size * max_seq_legth * hidden_size)
        """
        embedded_query = x * (token_type_ids.unsqueeze(dim=-1) == 0)
        embedded_query = self.dropout(F.relu(embedded_query))
        embedded_query = embedded_query[:, :30, :]  # b * 30 * dim
        embedded_query, _ = self.query_rnn(embedded_query)
        embedded_query = embedded_query[:, -1:, :]  # b * 1 * dimx2
        # embedded_query = embedded_query.reshape((x.shape[0], 1, -1))
        embedded_query = self.query_layer(embedded_query)

        mask = token_type_ids.unsqueeze(dim=-1) == 1
        mask[:, 0, :] = True
        embedded_key = x * mask
        embedded_key = self.key_layer(embedded_key)

        attention_rate = torch.matmul(
            embedded_key, torch.transpose(embedded_query, 1, 2)
        )
        attention_rate = attention_rate / math.sqrt(embedded_key.shape[-1])
        attention_rate = attention_rate / 10
        attention_rate = F.softmax(attention_rate, 1)

        embedded_value = self.value_layer(x)
        embedded_value = embedded_value * attention_rate
        if not self.need_query_emb:
            return embedded_value
        else:
            return embedded_value, embedded_query


class ConvLayer(nn.Module):
    """Conv layer for qa output"""

    def __init__(self, config):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=1,
        )

        self.conv3 = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding=1,
        )

        self.conv5 = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=5,
            padding=2,
        )

        self.drop_out = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Layer input
        Returns:
            torch.Tensor: output of conv layer (batch_size * hidden_size x 3 * max_seq_legth)
        """
        conv_input = x.transpose(1, 2)
        conv_output1 = F.relu(self.conv1(conv_input))
        conv_output3 = F.relu(self.conv3(conv_input))
        conv_output5 = F.relu(self.conv5(conv_input))
        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1)

        concat_output = concat_output.transpose(1, 2)
        concat_output = self.drop_out(concat_output)
        return concat_output


class QAConvSDSHead(nn.Module):
    """QA conv SDS head"""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        num_labels: int,
    ):
        """
        Args:
            input_size (int): max sequence lengths
            hidden_dim (int): backbone's hidden dimension
            n_layers (int): number of layers
            num_labels (int): number of labels used for QA
        """
        super().__init__()
        convs = []
        for n in range(n_layers):
            convs.append(QAConvSDSLayer(input_size, hidden_dim))
        self.convs = nn.Sequential(*convs)
        self.qa_output = nn.Linear(hidden_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape == (bsz, seq_length, hidden_dim)
        out = self.convs(x)
        return self.qa_output(out)  # (bsz, seq_length, hidden_dim)


class QAConvHeadWithAttention(nn.Module):
    """QA conv head with attention"""

    def __init__(self, config):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__()
        self.attention = AttentionLayer(config)
        self.conv = ConvLayer(config)
        self.classify_layer = nn.Linear(config.hidden_size * 3, 2, bias=True)

    def forward(self, x, token_type_ids):
        """
        Args:
            x (torch.Tensor): Head input
            token_type_ids (torch.Tensor): Token type ids of input_ids
        Returns:
            torch.Tensor: output logits (batch_size * max_seq_legth * 2)
        """
        embedded_value = self.attention(x, token_type_ids)
        concat_output = self.conv(embedded_value)
        logits = self.classify_layer(concat_output)
        return logits


class QAConvHead(nn.Module):
    """Simple QA conv head"""

    def __init__(self, config):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__()
        self.conv = ConvLayer(config)
        self.classify_layer = nn.Linear(config.hidden_size * 3, 2, bias=True)

    def forward(self, **kwargs):
        """
        Args:
            **kwargs: x, input_ids, sep_token_id
        Returns:
            torch.Tensor: output logits (batch_size * max_seq_legth * 2)
        """
        concat_output = self.conv(kwargs["x"])
        logits = self.classify_layer(concat_output)
        return logits
