import torch
import math

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)
x = 2.0
y = 4.0
lr = 0.1
z = w*x + b
a = torch.relu(z)
loss = (a - y)**2
loss.backward()
with torch.no_grad():
    w -= lr*w.grad
    b -= lr*b.grad
print("loss =", loss.item())
print("dw =", w.grad.item(), "db =", b.grad.item())
print("w_new =", w.item(), "b_new =", b.item())