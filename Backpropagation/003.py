
def relu_train_step(x, y, w, b, lr):
    z = w*x + b
    a = max(0, z)
    if (a > 0):
        dw = 2*(z - y)*x
        db = 2*(z - y)
    else:
        dw = 0
        db = 0
    loss = (a - y)**2
    w_new = w - lr*dw
    b_new = b - lr*db
    return loss, w_new, b_new

LossF, W, B = relu_train_step(x=2.0, y=4.0, w=-3.0, b=-1.0, lr=0.1) 
print ( LossF, " ", W, " ", B)