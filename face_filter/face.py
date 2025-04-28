import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image

# 이미지 로드
heart_img = cv2.imread('./heart.png', cv2.IMREAD_UNCHANGED)
heart_img = cv2.resize(heart_img, (30, 30))
tk_img = cv2.imread('./tk.png', cv2.IMREAD_UNCHANGED)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)


def overlay_png(background, overlay_img, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay_img.shape[:2]

    if x + ow > bw or y + oh > bh:
        return background  # 이미지 경계 초과 방지

    alpha_s = overlay_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        background[y:y+oh, x:x+ow, c] = (alpha_s * overlay_img[:, :, c] +
                                         alpha_l * background[y:y+oh, x:x+ow, c])
    return background


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.flip(frame, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = face_landmarks.landmark

            # 볼 좌표 평균
            c_left = [234, 93, 132]
            c_right = [454, 323, 361]
            lx = int(sum(w - landmarks[i].x * w for i in c_left) / 3)
            ly = int(sum(landmarks[i].y * h for i in c_left) / 3)
            rx = int(sum(w - landmarks[i].x * w for i in c_right) / 3)
            ry = int(sum(landmarks[i].y * h for i in c_right) / 3)

            # 오른쪽 눈 위 sparkle
            tk_right = face_landmarks.landmark[297]
            tk_x = int(w - tk_right.x * w)
            tk_y = int(tk_right.y * h) - 20

            # PNG 합성
            image = overlay_png(image, heart_img, lx - 20, ly - 20)
            image = overlay_png(image, heart_img, rx - 20, ry - 20)
            image = overlay_png(image, tk_img, tk_x - 50, tk_y - 15)

    cv2.imshow('FACE_Filter', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

