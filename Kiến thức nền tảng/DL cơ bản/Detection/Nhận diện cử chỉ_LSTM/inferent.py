import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

label = "Warmup...."
n_time_steps = 30
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model_new.h5")
cap = cv2.VideoCapture(0)
cap.set(3,1900)
cap.set(4,4000)

def draw_text(image,
              text,
              x,
              y,
              color_bgr=[255, 0, 0],
              size=0.05,  # in the range of (0, 1.0)
              font_face=cv2.FONT_HERSHEY_DUPLEX,
              thickness=0,  # 0: auto
              line_type=cv2.LINE_AA,
              is_copy=True):
    """
        Supported Fonts: https://docs.opencv.org/4.3.0/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
        Line Types: https://docs.opencv.org/4.3.0/d6/d6e/group__imgproc__draw.html#gaf076ef45de481ac96e0ab3dc2c29a777
    """
    assert size > 0

    image = image.copy() if is_copy else image  # copy/clone a new image
    if not text:  # empty text
        return image

    # https://docs.opencv.org/4.3.0/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82
    (text_width, text_height), _ = cv2.getTextSize(text, font_face, 1.0, thickness)  # estimate text size

    # calculate font scale
    h, w = image.shape[:2]
    short_edge = min(h, w)
    expect_size = short_edge * size
    font_scale = expect_size / text_height

    # calc thickness
    if thickness <= 0:
        thickness = int(font_scale)
        thickness = 1 if thickness == 0 else thickness

    # calc x,y in absolute coord
    x_abs = int(x * w)
    y_abs = int(y * h)

    # docs: https://docs.opencv.org/4.3.0/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    cv2.putText(img=image,
                text=text,
                org=(x_abs, y_abs),
                fontFace=font_face,
                fontScale=font_scale,
                color=color_bgr,
                thickness=thickness,
                lineType=line_type,
                bottomLeftOrigin=False)
    return image

def make_landmark_timestep(results):
    #print(results.pose_landmarks.landmark)
    c_lm = []
    landmark_ids = [ mpPose.PoseLandmark.NOSE,
                     mpPose.PoseLandmark.RIGHT_SHOULDER,
                     mpPose.PoseLandmark.LEFT_SHOULDER,
                     mpPose.PoseLandmark.LEFT_ELBOW,
                     mpPose.PoseLandmark.RIGHT_ELBOW,
                     mpPose.PoseLandmark.LEFT_WRIST,
                     mpPose.PoseLandmark.RIGHT_WRIST,
                     mpPose.PoseLandmark.LEFT_PINKY,
                     mpPose.PoseLandmark.RIGHT_PINKY,
                     mpPose.PoseLandmark.LEFT_INDEX,
                     mpPose.PoseLandmark.RIGHT_INDEX,
                     mpPose.PoseLandmark.LEFT_THUMB,
                     mpPose.PoseLandmark.RIGHT_THUMB,
                     mpPose.PoseLandmark.LEFT_HIP,
                     mpPose.PoseLandmark.RIGHT_HIP,
                     mpPose.PoseLandmark.LEFT_KNEE,
                     mpPose.PoseLandmark.RIGHT_KNEE,
                     mpPose.PoseLandmark.LEFT_ANKLE,
                     mpPose.PoseLandmark.RIGHT_ANKLE
                     ]
    for id in landmark_ids:
        lm = results.pose_landmarks.landmark[id]
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (255, 0, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
class_names = ['Body', 'Dumbbell Bicep Curl', 'Dumbbell Shoulder Press', 'Push Up', 'Belly Sticks']
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    #print(lm_list.shape)
    results = model.predict(lm_list)
    max_index = np.argmax(results)
    label = class_names[max_index]
    print('results:', np.max(results), label)

    return label

i = 0
warmup_frames = 60
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
stage_l = stage_r = ""
Stage_l = Stage_r = ""
while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        #print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list))
                t1.start()
                lm_list = []
            img = draw_landmark_on_image(mpDraw, results, img)

            #DUMBBELL BICEP CURL
            if results.pose_landmarks and label == 'Dumbbell Bicep Curl' and ((c_lm[11] > 0.7 and c_lm[15] > 0.7 and c_lm[23] > 0.7) or (c_lm[7] > 0.7 and c_lm[19] > 0.7 and c_lm[27] > 0.7)):

                shoulder_l = [c_lm[8], c_lm[9]]
                elbow_l = [c_lm[12], c_lm[13]]
                wrist_l = [c_lm[20], c_lm[21]]
                shoulder_r = [c_lm[4], c_lm[5]]
                elbow_r = [c_lm[16], c_lm[17]]
                wrist_r = [c_lm[24], c_lm[25]]
                angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                #LEFT
                if angle_l > 150:
                    stage_l = "down"
                if angle_l < 35 and stage_l == 'down':
                    stage_l = "up"
                    count_1 += 1
                    print(count_1)
                #RIGHT
                if angle_r > 150:
                    stage_r = "down"
                if angle_r < 35 and stage_r == 'down':
                    stage_r = "up"
                    count_2 += 1
                    print(count_2)

            #DUMBBELL SHOULDER PRESS
            if results.pose_landmarks and label == 'Dumbbell Shoulder Press' and ((c_lm[11] > 0.7 and c_lm[15] > 0.7 and c_lm[23] > 0.7) or (c_lm[7] > 0.7 and c_lm[19] > 0.7 and c_lm[27] > 0.7)):

                shoulder_l = [c_lm[8], c_lm[9]]
                elbow_l = [c_lm[12], c_lm[13]]
                wrist_l = [c_lm[20], c_lm[21]]
                shoulder_r = [c_lm[4], c_lm[5]]
                elbow_r = [c_lm[16], c_lm[17]]
                wrist_r = [c_lm[24], c_lm[25]]
                angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                #LEFT
                if angle_l < 35 and shoulder_l[1] < elbow_l[1]:
                    Stage_l = "down"
                if angle_l > 150 and shoulder_l[1] > elbow_l[1] and Stage_l == 'down':
                    Stage_l = "up"
                    count_3 += 1
                    print(count_3)
                print(shoulder_l[1], elbow_l[1])
                #RIGHT
                if angle_r < 35 and shoulder_r[1] < elbow_r[1]:
                    Stage_r = "down"
                if angle_r >150 and shoulder_r[1] > elbow_r[1] and Stage_r == 'down':
                    Stage_r = "up"
                    count_4 += 1
                    print(count_4)

            # Push Up
            if results.pose_landmarks and label == 'Push Up' and (
                    (c_lm[11] > 0.7 and c_lm[15] > 0.7 and c_lm[23] > 0.7) or (
                    c_lm[7] > 0.7 and c_lm[19] > 0.7 and c_lm[27] > 0.7)):

                shoulder_l = [c_lm[8], c_lm[9]]
                elbow_l = [c_lm[12], c_lm[13]]
                wrist_l = [c_lm[20], c_lm[21]]
                angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                # LEFT
                if angle_l < 35 and shoulder_l[1] < elbow_l[1]:
                    Stage_l = "down"
                if angle_l > 150 and shoulder_l[1] > elbow_l[1] and Stage_l == 'down':
                    Stage_l = "up"
                    count_5 += 1

    img = cv2.flip(img, 1)
    img = draw_text(image=img,
                      text="Bicep Cur:        LEFT: " + str(count_1) + "   " + "RIGHT:" + str(count_2),
                      x=0,
                      y=0.06,
                      size=0.025,
                      color_bgr=[0, 0 , 255],
                      is_copy=True)

    img = draw_text(image=img,
                    text="Shoulder Press:   LEFT: " + str(count_3) + "   " + "RIGHT:" + str(count_4),
                    x=0,
                    y=0.12,
                    size=0.025,
                    color_bgr=[0, 0, 255],
                    is_copy=True)

    img = draw_text(image=img,
                    text="Push Up:" +  str(count_5),
                    x=0,
                    y=0.18,
                    size=0.025,
                    color_bgr=[0, 0, 255],
                    is_copy=True)
    '''''''''
    cv2.putText(img, str(angle),
                tuple(np.multiply(elbow_l, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )
    '''''''''
    img = draw_class_on_image(label, img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()