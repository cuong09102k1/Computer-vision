# Đọc dữ liệu từ file csv vào DataFrame
import pandas as pd
df = pd.read_csv('Body.txt')

import matplotlib.pyplot as plt
# Tạo biểu đồ với 3 đường
plt.plot(df['20'], color='red', linewidth=2)
plt.plot(df['21'], color='green', linewidth=2)
#plt.plot(df['24'], color='blue', linewidth=2)
#plt.plot(df['25'], color='black', linewidth=2)

# Thêm tiêu đề và nhãn
plt.title('Biểu đồ 2 chiều với 3 đường')
plt.xlabel('Trục x')
plt.ylabel('Trục y')

# Hiển thị đồ thị
plt.show()