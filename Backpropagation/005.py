import torch
import math

def training(X, Y):
    w = torch.zeros((len(X[0]),1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.0001
    for i in range(1, 1000001):
        y_pred = X @ w + b
        loss_Func = ((y_pred - Y)**2).mean()
        loss_Func.backward()
        with torch.no_grad():
            w -= lr*w.grad
            b -= lr*b.grad
        if i%20000 == 0: 
            print(f"Epoch {i}: loss={loss_Func.item():.10f}, w={w.view(-1).tolist()}, b={b.item():.10f}")
        w.grad.zero_()
        b.grad.zero_()
    return loss_Func, w, b

X = torch.tensor([
    [ 50.0, 2.0],
    [ 80.0, 3.0],
    [100.0, 3.0],
    [120.0, 4.0],
    [ 60.0, 2.0],
    [150.0, 5.0],
])  # cột 1 lớn (50..150), cột 2 nhỏ (2..5)
Y = torch.tensor([[7.5], [11.0], [12.0], [15.0], [8.0], [18.5]])
if __name__ == "__main__":
    loss, w, b = training(X, Y)