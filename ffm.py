import time
import torch

from torch import nn


batch_size = 1024
category_feature_size = 10
embedding_size = 100
v_size = 50
output_size = 100

input = torch.rand([batch_size, category_feature_size, embedding_size])
input = torch.flatten(input, start_dim=1)
field_feature_sizes = [embedding_size for _ in range(category_feature_size)]

field_size = len(field_feature_sizes)
feature_size = sum(field_feature_sizes)  # feature_size * embedding_size
linear = nn.Linear(feature_size, output_size)  # [output_size, feature_size] * [feature_size]
v_embedding = nn.Parameter(torch.randn(feature_size, field_size, v_size, output_size))
feature_field_mapping = {}
offset = -1
offsets = []
for field_index in range(field_size):
    offsets.append(offset + 1)  # start_index
    for _ in range(field_feature_sizes[field_index]):
        offset += 1
        feature_field_mapping[offset] = field_index
offsets.append(feature_size)

first_order_term = linear(input)
print(first_order_term)

time1 = time.time()
second_order_cross_term = 0
for i in range(feature_size - 1):
    for j in range(i + 1, feature_size):
        weights = torch.sum(torch.mul(v_embedding[i, feature_field_mapping[j], :], v_embedding[j, feature_field_mapping[i], :]), dim=0)
        xx = torch.mul(input[:, i], input[:, j])
        second_order_cross_term += torch.ger(xx, weights)
time2 = time.time()
second_order_cross_term_ = 0
for i in range(field_size):
    for j in range(i, field_size):
        field_cross_size = field_feature_sizes[i] * field_feature_sizes[j]
        weights = torch.sum(torch.einsum('nij,mij->nmij', v_embedding[offsets[i]: offsets[i + 1], i, :, :], v_embedding[offsets[j]: offsets[j + 1], j, :, :]).view(field_cross_size, v_size, output_size), dim=1)
        xx = torch.einsum('ni,nj->nij', input[:, offsets[i]: offsets[i + 1]], input[:, offsets[j]: offsets[j + 1]]).view(-1, field_cross_size)
        current_term = torch.matmul(xx, weights)
        if i == j:
            diagonal_term = torch.matmul(input[:, offsets[i]: offsets[i + 1]].pow(2), v_embedding[offsets[i]: offsets[i + 1], i, :, :].pow(2).sum(dim=1))
            current_term -= diagonal_term
            current_term /= 2
        second_order_cross_term_ += current_term
time3 = time.time()
print(second_order_cross_term_)

fm = first_order_term + second_order_cross_term
print(fm)


print('method1 cost', time2 - time1, 's')
print('method2 cost', time3 - time2,'s')

# input = [batch_size, feature_size]
# v = [field_size, feature_size, v_size, output_size]
# => [batch_size, output_size]
#
# input * input = [batch_size, feature_size, feature_size]
# v * v => [feature_size, feature_size, output_size]
# => [batch_size, output_size]