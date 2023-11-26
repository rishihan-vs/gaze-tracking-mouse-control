import dlib
import cv2
import numpy as np
import imutils
from imutils import face_utils

video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# image = cv2.imread("face1.jpeg")
# image = imutils.resize(image, width=500)

while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # cv2.putText(frame, f"Face #{i+1}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            for (x, y) in shape[i:j]:
                if name == "mouth" or name == "inner_mouth" or name == "left_eye" or name == "right_eye":
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)

    cv2.imshow("Landmarks", frame)
    key = cv2.waitKey(1)
    if key == 27:           # ESC key
        break

video.release()
cv2.destroyAllWindows()