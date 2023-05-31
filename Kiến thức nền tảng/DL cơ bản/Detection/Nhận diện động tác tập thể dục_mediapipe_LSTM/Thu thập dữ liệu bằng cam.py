import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
lm_list = []
IMAGE_FILES = []
label = "Body_new"
BG_COLOR = (192, 192, 192) # gray

def make_landmark_timestep(results):
    c_lm = []
    landmark_ids = [mp_pose.PoseLandmark.NOSE,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.LEFT_PINKY,
                    mp_pose.PoseLandmark.RIGHT_PINKY,
                    mp_pose.PoseLandmark.LEFT_INDEX,
                    mp_pose.PoseLandmark.RIGHT_INDEX,
                    mp_pose.PoseLandmark.LEFT_THUMB,
                    mp_pose.PoseLandmark.RIGHT_THUMB,
                    mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.LEFT_KNEE,
                    mp_pose.PoseLandmark.RIGHT_KNEE,
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE
                    ]
    for id in landmark_ids:
        lm = results.pose_landmarks.landmark[id]
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

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


with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video kích thước {width}x{height}")
i = -150
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    i = i + 1
    print(i)
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    image = cv2.flip(image, 1)
    if i < 0:
        image = draw_text(image=image,
                          text="Coming soon...(i >= 0: i = " + str(i)+ ")",
                          x=0,
                          y=0.08,
                          size=0.05,
                          color_bgr=[0, 0, 0],
                          is_copy=True)

    if i > 0 and i < 9000:
        image = draw_text(image=image,
                          text="Loading...(frame =  " + str(i) + ")",
                          x=0,
                          y=0.08,
                          size=0.05,
                          color_bgr=[0, 0, 0],
                          is_copy=True)

        if results.pose_landmarks:
            # ghi nhan thong so
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

import pandas as pd
#luu vao csv
df = pd.DataFrame(lm_list)
#df.to_csv(label + ".txt")
csv_path = label + ".txt"
df.to_csv(csv_path, mode='a', header=False, index=False)
cap.release()
cv2.destroyAllWindows()