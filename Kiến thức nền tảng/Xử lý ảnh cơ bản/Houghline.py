import cv2
import numpy as np

# Đọc ảnh đầu vào và chuyển sang ảnh xám
img = cv2.imread('anh2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Áp dụng Gaussian Blur để làm mịn ảnh
gray = cv2.GaussianBlur(gray,(5,5),0)

# Sử dụng Canny edge detection để tìm cạnh
edges = cv2.Canny(gray,50,150,apertureSize = 3)

# Áp dụng Hough Line Transform để tìm các đường thẳng
lines = cv2.HoughLines(edges,1,np.pi/180,200)

# Vẽ các đường thẳng tìm được lên ảnh gốc
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Hiển thị ảnh gốc và ảnh đã tìm được các đường thẳng
cv2.imshow('input',img)
cv2.imshow('lines',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()