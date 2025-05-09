import os
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose

# if you are using colab
#from google.colab.patches import cv2_imshow

# PyTorch Hub
import torch

# Model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# since we are only intrested in detecting person
yolo_model.classes = [0]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = "D:\\Drone\\fallen.mp4"

# get the dimension of the video
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    h, w, _ = frame.shape
    size = (w, h)
    print(size)
    break

cap = cv2.VideoCapture(video_path)

# for webacam cv2.VideoCapture(NUM) NUM -> 0,1,2 for primary and secondary webcams..

# For saving the video file as output.avi
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, size)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # making image writeable to false improves prediction
    image.flags.writeable = False

    result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(result.xyxy)  # img1 predictions (tensor)

    # This array will contain crops of images incase we need it
    img_list = []

    # we need some extra margin bounding box for human crops to be properly detected
    MARGIN = 10

    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            # Media pose prediction ,we are
            results = pose.process(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:])

            # Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing
            mp_drawing.draw_landmarks(
                image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            img_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax):])
    # cv2_imshow(image)

    # writing in the video file
    out.write(image)

    ## Code to quit the video incase you are using the webcam
    cv2.imshow('Activity recognition', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
         break

cap.release()
out.release()
cv2.destroyAllWindows()