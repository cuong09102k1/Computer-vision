import cv2
import numpy as np

img = cv2.imread('anh.png', cv2.IMREAD_UNCHANGED)

#convert img to grey
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#set a thresh(giá trị ngưỡng, tất cả các pixel có giá trị lớn hơn 100 sẽ là 255(là tham số được gán ở đầu vào hàm) và nhỏ hơn sẽ là 0)
thresh = 125
gray_blur = cv2.GaussianBlur(img_grey, (5, 5), 0)
#get threshold image
ret,thresh_img = cv2.threshold(gray_blur, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Vẽ bounding box cho từng contour
for cnt in contours:
    # Lấy thông tin của bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    # Vẽ bounding box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # In ra thông số của bounding box
    print("Bounding box: x=", x, " y=", y, " width=", w, " height=", h)
#create an empty image for contours
#img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv2.drawContours(img, contours, -1, (0,255,0), 3)
print("Number of objects detected: ", len(contours))#in ra số lượng vật thể
#save image
cv2.imshow('images', img)
cv2.waitKey()
cv2.destroyAllWindows()