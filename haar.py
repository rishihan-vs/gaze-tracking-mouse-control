import cv2
import numpy as np

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_detect = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')

image = cv2.imread('face1.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_detect.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in face:
    image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

roi_gray = gray[y:y+h, x:x+w]           # cropped detected face in graysacle
roi_colour = image[y:y+h, x:x+w]         # cropped detected face in colour

mouth = mouth_detect.detectMultiScale(roi_gray)
for (mx, my, mw, mh) in mouth:
    cv2.rectangle(roi_colour, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)

eyes = eye_detect.detectMultiScale(roi_gray)
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(roi_colour, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()