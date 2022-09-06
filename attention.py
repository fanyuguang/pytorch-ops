#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn.functional as F

from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['entity_num_attention_heads']
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x):
        '''
        Input: (2, 512, 768)
        Output: (2, 12, 512, 64)
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(input))  # Output: (2, 12, 512, 64)
        key_layer = self.transpose_for_scores(self.key(input))  # Output: (2, 12, 512, 64)
        value_layer = self.transpose_for_scores(self.value(input))  # Output: (2, 12, 512, 64)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # Output: (2, 12, 512, 512)
        attention_scores = attention_scores / math.sqrt(self.hidden_size)  # Output: (2, 12, 512, 512)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # Output: (2, 12, 512, 512)
        attention_probs = self.dropout(attention_probs)

        outputs = torch.matmul(attention_probs, value_layer)  # Output: (2, 12, 512, 64)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()  # Output: (2, 512, 12, 64)
        new_outputs_shape = outputs.size()[:-2] + (self.all_head_size,)  # Output: (2, 512, 768)
        outputs = outputs.view(new_outputs_shape)  # Output: (2, 512, 768)
        return outputs



config = {
    'hidden_size': 768,
    'entity_num_attention_heads': 12,
    'attention_probs_dropout_prob': 0.5
}
attention = SelfAttention(config)

input = torch.randn(2, 512, 768)  # shape (batch_size, seq_len:512, hidden_size:768)
outputs = attention(input)
print(outputs)

