import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# load dữ liệu, chuyển về dạng numpy
data = pd.read_csv('data_linear.csv').values

# chuyển dữ liệu về dạng torch
x = torch.tensor(data[:,0])
y = torch.tensor(data[:,1])
print(x,y)

# hàm model f(x) = ax+b
def model(x, a, b):
    return a * x +b

# hàm loss function, Mean Squared Error
def loss_fn(y_hat, y):
    squared_diffs = 0.5*(y_hat - y)**2
    return squared_diffs.mean() #hàm mean() sẽ tính trung bình của tất cả các phần tử trong tensor


# Hàm training
def training_loop(n_epochs, learning_rate, params, x, y):
    a, b = params
    # Lưu loss qua epoch để vẽ đồ thị loss
    losses = []
    for epoch in range(1, n_epochs + 1):
        # nếu có grad ở tham số a, b thì zero đi, tránh trường hợp cộng dồn grad
        if a.grad is not None:
            a.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()

        # xây model, loss
        y_hat = model(x, a, b)
        loss = loss_fn(y_hat, y)

        # gọi backward để tính đạo hàm ngược của loss với tham số a, b
        loss.backward()

        # update a,b bằng thuật toán gradient descent, để torch.no_grad thì mình không cần backward ở bước này
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad

        if epoch % 1 == 0:
            losses.append(loss.item())
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return a, b, losses


a = torch.ones((),requires_grad=True)
b = torch.zeros((),requires_grad=True)

a, b, losses = training_loop(30, 0.00005, (a, b), x, y)

# Dự đoán giá trị mới, x = 50
x = torch.tensor(50)

#Để tránh việc tốn bộ nhớ khi thực hiện những phép tính không cần grad
with torch.no_grad():
    y_hat = model(x, a, b)
    print(y_hat)

print(a,b)

x = torch.tensor(data[:,0])
y = torch.tensor(data[:,1])
with torch.no_grad():
    y_hat = model(x, a, b)
plt.scatter(x, y)
plt.plot(x, y_hat, c='r')
plt.show()


