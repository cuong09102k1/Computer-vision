#Trong sharpen, kernel, ta bắt đầu thấy các hệ số giá trị âm xen kẽ hệ số dương mục đích để đào sâu sự khác biệt điểm ảnh chính giữa với các điểm ảnh xung quanh.
import numpy as np
import cv2
kernel3 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
image = '1522_Thuy_Ngan_Photo_VieON_3.jpg'
image = cv2.imread(image)
sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel3)

#cv2.imshow('Original', image)
cv2.imshow('Sharpened', sharp_img)

cv2.waitKey()
cv2.destroyAllWindows()