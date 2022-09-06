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
feature_size = category_feature_size * embedding_size

linear = nn.Linear(feature_size, output_size)
v_embedding = nn.Parameter(torch.randn(feature_size, v_size * output_size))


first_order_term = linear(input)
print(first_order_term)

time1 = time.time()
square_of_sum = torch.matmul(input, v_embedding).pow(2).view(-1, v_size, output_size)
sum_of_square = torch.matmul(input.pow(2), v_embedding.pow(2)).view(-1, v_size, output_size)
second_order_cross_term = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
time2 = time.time()
print(second_order_cross_term)

fm = first_order_term + second_order_cross_term
print(fm)

print('time cost', time2 - time1, 's')