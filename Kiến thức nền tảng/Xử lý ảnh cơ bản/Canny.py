import cv2
import numpy as np

# Đọc ảnh đầu vào
img = cv2.imread('anh.png')

# Chuyển ảnh sang độ xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Áp dụng bộ lọc Gaussian để giảm nhiễu
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# Sử dụng Canny edge detection để tìm cạnh
edges = cv2.Canny(gaussian, 50, 150)
#50: ngưỡng dưới. Tất cả các giá trị gradient thấp hơn ngưỡng này sẽ bị loại bỏ.
#150: ngưỡng trên. Tất cả các giá trị gradient cao hơn ngưỡng này sẽ được giữ lại.

# Hiển thị ảnh kết quả
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
