import cv2
import numpy as np
import matplotlib.pyplot as plt


# You can set custom kernel size if you want.
kernel = None

# load a video
#cap = cv2.VideoCapture('media/videos/vtest.avi')

# you can optionally work on the live webcam
cap = cv2.VideoCapture('video_oto.mp4')

# tạo một đối tượng phân đoạn nền động(Nó hoạt động bằng cách giữ lại bức ảnh nền của một video và loại bỏ phần nền đó để phân tách các đối tượng chuyển động.)
backgroundobject = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
#giá trị history sẽ xác định số lượng khung hình liên tiếp được sử dụng để ước tính nền, từ đó xác định các pixel nào là nền và các pixel nào là đối tượng trong khung hình.
while (1):
    ret, frame = cap.read()
    if not ret:
        break

    #Hàm này nhận vào một frame và trả về mặt nạ nhị phân của các vật thể di chuyển so với nền.
    fgmask = backgroundobject.apply(frame)

    # loại bỏ các pixel có độ sáng thấp hơn ngưỡng 250, đồng thời chuyển tất cả các pixel có giá trị lớn hơn ngưỡng này thành 255, giúp mặt nạ trở nên rõ ràng hơn.
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # các phép toán hình thái học như erode và dilate để loại bỏ nhiễu và các lỗ thủng nhỏ trên mặt nạ.
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # Detect contours in the frame.
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the frame to draw bounding boxes around the detected cars.
    frameCopy = frame.copy()

    # loop over each contour found in the frame.
    for cnt in contours:

        # Make sure the contour area is somewhat higher than some threshold to make sure its a car and not some noise.
        if cv2.contourArea(cnt) > 15000:
            # Retrieve the bounding box coordinates from the contour.
            x, y, width, height = cv2.boundingRect(cnt)

            # Draw a bounding box around the car.
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)

            # Write Car Detected near the bounding box drawn.
            cv2.putText(frameCopy, 'Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)
            #print('width:',width)
            #print('height:',height)
            print(cv2.contourArea(cnt))

    # trích xuất phần của frame gốc frame mà được phân đoạn từ fgmask
    real_part = cv2.bitwise_and(frame, frame, mask=fgmask)

    # chuyển đổi mặt nạ nhị phân fgmask từ kênh màu xám sang kênh màu BGR
    # fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

    stacked = np.hstack((frame, real_part, frameCopy))
    cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked, None, fx=0.3, fy=0.3))
    #giam FPS(voi 20, ta co FPS con 50FPS)
    #cv2.waitKey(20)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()