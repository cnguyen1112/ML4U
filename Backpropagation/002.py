X = [1.0, 2.0, 3.0, 4.0]
Y = [3.0, 5.0, 7.0, 9.0]
w = 0.0
b = 0.0
lr = 0.01

def cost(X, w, b):
    return [w*X[i] + b for i in range(len(X))]

def loss(Y_train,Y):
    sub = [(Y_train[i] - Y[i])**2 for i in range(len(Y))]
    loss_func = (1/(len(Y)))*sum(sub)
    return loss_func

def grad_w(X, Y_train, Y):
    sub = [(Y_train[i] - Y[i])*2*X[i] for i in range(len(Y))]
    grad_w_func = (1/(len(Y)))*sum(sub)
    return grad_w_func

def grad_b(Y_train, Y):
    sub = [(Y_train[i] - Y[i])*2 for i in range(len(Y))]
    grad_b_func = (1/(len(Y)))*sum(sub)
    return grad_b_func

def training_func(X, Y, w, b, lr):
    for i in range(1, 10001):
        Y_train = cost(X, w, b)
        dw = grad_w(X, Y_train, Y)
        db = grad_b(Y_train, Y)
        lossF = loss(Y_train,Y)
        w -= lr*dw
        b -= lr*db
        if (i % 100 == 0):
            print("Weight: ",w, " Bias: ", b, " Loss:", lossF)
    return w,b
