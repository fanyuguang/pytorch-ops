#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def main():
    model = LinearModel()
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x_train_data = torch.Tensor([[float(i)] for i in range(1, 3)])
    y_train_data = torch.Tensor([[float(i * 2)] for i in range(1, 3)])
    
    if torch.cuda.is_available():
        print('cuda is available')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    x_train_data = x_train_data.to(device)
    y_train_data = y_train_data.to(device)
    
    for epoch in range(1000):
        y_train_pred = model(x_train_data)
        loss = criterion(y_train_pred, y_train_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch, loss.item())
    
    x_test_data = torch.Tensor([[4.0]])
    x_test_data = x_test_data.to(device)
    y_test_pred = model(x_test_data)
    print('x_test_data = ', x_test_data.item())
    print('y_test_pred = ', y_test_pred.data)


if __name__ == '__main__':
    main()

