import cv2
import dlib 
from imutils import face_utils, resize 
import numpy as np

orange_img = cv2.imread('./orange.png') 
orange_img = cv2.resize(orange_img, dsize=(512, 512))

detector = dlib.get_frontal_face_detector() # dlib의 얼굴 영역 탐지
# 랜드마크 탐지
predictor = dlib.shape_predictor("C:/dlib-19.24/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read() # read 로 이미지를 읽어준 다음에

    if not ret: # 프레임이 없으면 반복문 빠져 나옴
        break

    faces = detector(img) # 얼굴영역을 인식해주면 faces에 좌표 정보가 들어간다.

    result = orange_img.copy() # result에 오렌지 이미지를 카피. 

    if len(faces) > 0: # 얼굴이 1개 이상이면 실행
        face = faces[0] # 1개의 얼굴만 사용하므로 0번 인덱스만 face라는 변수에 저장
        
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy() # face_img에 저장을 해준다

        shape = predictor(img, face) # 랜드마크 68개의 점 구하기
        shape = face_utils.shape_to_np(shape) # dlib 오브젝트를 numpy로 바꿔준다.

        for p in shape:
            cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        ## eyes
        le_x1 = shape[36, 0] #왼쪽 눈끝
        le_y1 = shape[37, 1] 
        le_x2 = shape[39, 0] #왼쪽 눈 앞머리
        le_y2 = shape[41, 1]
        le_margin = int((le_x2 - le_x1) * 0.2) #픽셀값: 개인 크기에 맞춰서 변경

        re_x1 = shape[42, 0] #오른족 눈 앞머리
        re_y1 = shape[43, 1]
        re_x2 = shape[45, 0] #오른쪽 눈 끝
        re_y2 = shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.2)
        #크롭 후 눈 이미지 저장
        left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy()
        right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()

        left_eye_img = resize(left_eye_img, width=100) # 가로 100 resize
        right_eye_img = resize(right_eye_img, width=100)

        # 왼쪽눈 합성
        result = cv2.seamlessClone(
            left_eye_img, 
            result, 
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            (180, 227),
            cv2.MIXED_CLONE
        )
        # 오른쪽 눈 합성
        result = cv2.seamlessClone(
            right_eye_img, 
            result,
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            (310, 227),
            cv2.MIXED_CLONE
        )

        ## mouth
        mouth_x1 = shape[48, 0] #왼쪽 입 꼬리
        mouth_y1 = shape[52, 1] #입술 위
        mouth_x2 = shape[54, 0] #오른쪽 입꼬리
        mouth_y2 = shape[57, 1] #입술 끝
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1) #픽셀값
        
        # 크롭해서 입 이미지 저장
        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        mouth_img = resize(mouth_img, width=250)

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (245, 309),
            cv2.MIXED_CLONE
        )

        # cv2.imshow('left', left_eye_img)
        # cv2.imshow('right', right_eye_img)
        # cv2.imshow('mouth', mouth_img)
        cv2.imshow('face', face_img)

        #이미지의 좌표값 생성
        def mouse_callback(event, x, y, flags, param):
             if event == cv2.EVENT_LBUTTONDOWN:
                  print(f"클릭한 좌표: ({x}, {y})")

        cv2.namedWindow("result")
        cv2.setMouseCallback("result", mouse_callback)

        cv2.imshow('result', result)

    # cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break