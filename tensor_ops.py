import torch


x = torch.randint(1, 9, [2])
y = torch.randint(1, 9, [2])
res = torch.dot(x, y)
print(res.size())

x = torch.randint(1, 9, [2])
y = torch.randint(1, 9, [3])
res = torch.ger(x, y)
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [3])
res = torch.mv(x, y)
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [2, 3])
res = torch.mul(x, y)
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [3, 4])
res = torch.mm(x, y)
print(res.size())

x = torch.randint(1, 9, [5, 2, 3])
y = torch.randint(1, 9, [5, 3, 4])
res = torch.bmm(x, y)
print(res.size())

############# torch.matmul ###############

x = torch.randint(1, 9, [2])
y = torch.randint(1, 9, [2])
res = torch.matmul(x, y)  # same with torch.dot()
print(res.size())

x = torch.randint(1, 9, [2])
y = torch.randint(1, 9, [2, 3])
res = torch.matmul(x, y)
print(res.size())

x = torch.randint(1, 9, [3])
y = torch.randint(1, 9, [2, 3, 4])
res = torch.matmul(x, y)
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [3])
res = torch.matmul(x, y)  # same with torch.mv()
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [3, 4])
res = torch.matmul(x, y)  # same with torch.mm()
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [5, 3, 4])
res = torch.matmul(x, y)
print(res.size())

x = torch.randint(1, 9, [5, 2, 3])
y = torch.randint(1, 9, [3])
res = torch.matmul(x, y)
print(res.size())

x = torch.randint(1, 9, [5, 2, 3])
y = torch.randint(1, 9, [3, 4])
res = torch.matmul(x, y)
print(res.size())

x = torch.randint(1, 9, [5, 2, 3])
y = torch.randint(1, 9, [5, 3, 4])
res = torch.matmul(x, y)  # same with torch.bmm()
print(res.size())

x = torch.randint(1, 9, [5, 1, 2, 3])
y = torch.randint(1, 9, [6, 3, 4])
res = torch.matmul(x, y)
print(res.size())

############# torch.einsum ###############

x = torch.randint(1, 9, [2, 3])
res = torch.einsum('ij->', x)  # same with torch.sum(x)
print(res.size())

x = torch.randint(1, 9, [2, 3])
res = torch.einsum('ij->j', x)  # same with torch.sum(x, dim=0)
print(res.size())

x = torch.randint(1, 9, [2, 3])
res = torch.einsum('ij->i', x)  # same with torch.sum(x, dim=1)
print(res.size())

x = torch.randint(1, 9, [2, 3])
res = torch.einsum('ij->ji', x)  # same with torch.transpose(x)
print(res.size())

x = torch.randint(1, 9, [2])
y = torch.randint(1, 9, [2])
res = torch.einsum('i,i->', x, y)  # same with torch.dot(x, y)
print(res.size())

x = torch.randint(1, 9, [2])
y = torch.randint(1, 9, [3])
res = torch.einsum('i,j->ij', x, y)  # same with torch.ger(x, y)
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [3])
res = torch.einsum('ij,j->i', x, y)  # same with torch.mv(x,y)
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [2, 3])
res = torch.einsum('ij,ij->ij', x, y)  # same with torch.mul(x, y)
print(res.size())

x = torch.randint(1, 9, [2, 3])
y = torch.randint(1, 9, [3, 4])
res = torch.einsum('ij,jk->ik', x, y)  # same with torch.mm(x, y)
print(res.size())

x = torch.randint(1, 9, [5, 2, 3])
y = torch.randint(1, 9, [5, 3, 4])
res = torch.einsum('bij,bjk->bik', x, y)  # same with torch.bmm(x, y)
print(res.size())
