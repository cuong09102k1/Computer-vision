import cv2
import mediapipe as mp
import time
import pyautogui, sys
import math
from PIL import Image


# class creation
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    # Draw dots and connect them
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            # Which hand are we talking about
            myHand = self.results.multi_hand_landmarks[handNo]
            # Get id number and landmark information
            for id, lm in enumerate(myHand.landmark):
                # id will give id of landmark in exact index number
                # height width and channel
                h, w, c = img.shape
                # find the position
                cx, cy = int(lm.x * w), int(lm.y * h)  # center
                # print(id,cx,cy)
                lmlist.append([id, cx, cy])

                # Draw circle for 0th landmark
                if draw:
                    cv2.circle(img, (cx, cy), 1 , (0, 0, 1), cv2.FILLED)
        return lmlist

    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    # Frame rates
    width, height = 1280, 720
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    cap.set(3, width)
    cap.set(4, height)
    pTime = 0
    cTime = 0
    count = 0
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[8], lmList[11], lmList[4], lmList[5])
            ax = lmList[8][1]
            ay = lmList[8][2]
            bx = lmList[11][1]
            by = lmList[11][2]
            dx = lmList[4][1]
            dy = lmList[4][2]
            ex = lmList[5][1]
            ey = lmList[5][2]
            #dieu khien chuot bang diem thu 8(dau ngon tro)
            mx = lmList[8][1] - 160
            my = lmList[8][2] - 20
            if (mx <=0 and my <=0):
                pyautogui.moveTo(1919,1)
            if (mx >= 960 and my <= 0):
                pyautogui.moveTo(1, 1)
            if (mx <=0 and my >= 540):
                pyautogui.moveTo(1919,1079)
            if (mx >= 960 and my >= 540):
                pyautogui.moveTo(1, 1079)
            if ((mx>0 or my >0) and (mx <960 or my >0) and(mx>0 or my <540)and(mx <960 or my <540)):
                pyautogui.moveTo(1920-mx*2,my*2)
                if math.sqrt((dx - ex) * (dx - ex) + (dy - ey) * (dy - ey)) <= 40:
                   pyautogui.click(button='left')
                   time.sleep(1)
            #thao tac chon vung phong to slide
            if math.sqrt((dx - ax) * (dx - ax) + (dy - ay) * (dy - ay)) <= 40:
                pyautogui.moveTo(220, 1054)
                pyautogui.click(button='left')
                time.sleep(1)
                count = 1
            # thao tac lui slide va thoat phong to
            if math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by)) <= 50:
                if count == 0:
                   pyautogui.moveTo(22, 1054)
                   pyautogui.click(button='left')
                   time.sleep(1)
                if count == 1:
                   pyautogui.click(button='right')
                   time.sleep(1)
                   count = 0



        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        img1 = cv2.flip(img, 0)
        img2 = cv2.rotate(img1,cv2.cv2.ROTATE_180)
        cv2.putText(img2, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        screen = cv2.rectangle(img2,(160,20),(width-160,height-160),(255,255,255),5)

        # Shows the image in image viewer
        #im1.show()
        cv2.imshow("Video", screen)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()