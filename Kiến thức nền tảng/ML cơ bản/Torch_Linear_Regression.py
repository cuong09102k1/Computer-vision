import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# load dữ liệu, chuyển về dạng numpy
data = pd.read_csv('data_linear.csv').values

# chuyển dữ liệu về dạng torch
#tensor "x" và "y" ban đầu có kích thước là (n,)
x = torch.tensor(data[:,0], dtype=torch.float32)
y = torch.tensor(data[:,1], dtype=torch.float32)
#"unsqueeze(1)" để thêm một chiều mới vào tensor "x" và "y". Khi đó, kích thước của chúng sẽ trở thành (n, 1)
x = x.unsqueeze(1)
y = y.unsqueeze(1)

# Hàm training
def training_loop(n_epochs, optimizer, model, loss_fn, x, y):
    losses = []
    for epoch in range(1, n_epochs + 1):
      y_hat = model(x)
      print(y_hat.shape, y.shape)
      loss = loss_fn(y_hat, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if epoch % 1 == 0:
              losses.append(loss.item())
              print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return losses
#định nghĩa một mô hình học máy đơn giản là một mạng neural tuyến tính với một đầu vào và một đầu ra
linear_model = nn.Linear(1, 1)
#sử dụng thuật toán tối ưu hóa SGD (Stochastic Gradient Descent) để cập nhật các tham số của mô hình trong quá trình huấn luyện
optimizer = optim.SGD(linear_model.parameters(), lr=0.00004)
#với đầu vào là danh sách các tham số cần cập nhật trong mô hình, được lấy ra bằng phương thức "linear_model.parameters()", và tỷ lệ học (learning rate) lr được đặt là 0.00004.

#lấy được danh sách các tham số của mô hình tuyến tính, gồm hai giá trị là weight và bias
list(linear_model.parameters())

loss = training_loop(
  n_epochs = 10,
  optimizer = optimizer,
  model = linear_model,
  loss_fn = nn.MSELoss(),
  x = x,
  y = y
)

plt.plot(loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# chuyển dữ liệu về dạng torch
x = torch.tensor(data[:,0], dtype=torch.float32)
y = torch.tensor(data[:,1], dtype=torch.float32)
x = x.unsqueeze(1)

with torch.no_grad():
    y_hat = linear_model(x)
plt.scatter(x, y)
plt.plot(x, y_hat, c='r')
plt.show()

