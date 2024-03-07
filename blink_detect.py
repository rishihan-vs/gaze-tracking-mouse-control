import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from imutils.video import VideoStream
from imutils import face_utils
import datetime
from threading import Thread

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()
      
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
		
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
      

def getWebcamFramerate(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
 
    # Number of frames to capture
    num_frames = 120
 
    print("Capturing {0} frames".format(num_frames))
 
    # Start time
    start = time.time()
 
    # Grab a few frames
    for i in range(0, num_frames) :
        ret, frame = video.read()
 
    # End time
    end = time.time()
 
    # Time elapsed
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))
 
    # Calculate frames per second
    fps  = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	eyeAR = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return eyeAR

def mouth_aspect_ratio(mouth):
     A = dist.euclidean(mouth[2], mouth[7])
     B = dist.euclidean(mouth[3], mouth[5])

     C = dist.euclidean(mouth[0], mouth[4])

     mouthAR = (A + B) / (2.0 * C)

     return mouthAR


EYE_AR_THRESHOLD = 0.21
MOUTH_AR_THRESHOLD = 0.35
EYE_AR_CONSEC_FRAMES = 3

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# getWebcamFramerate(video)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # cv2.putText(frame, f"Face #{i+1}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            for (x, y) in shape[i:j]:
                if name == "mouth" or name == "inner_mouth" or name == "left_eye" or name == "right_eye":
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        mouth = shape[mStart:mEnd]
        mouthAR = mouth_aspect_ratio(mouth)

        cv2.putText(frame, "LEFT EAR: {}".format(leftEAR), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "RIGHT EAR: {}".format(rightEAR), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MOUTH: {}".format(mouthAR), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        if leftEAR < EYE_AR_THRESHOLD:
            cv2.putText(frame, "LEFT EYE BLINK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if rightEAR < EYE_AR_THRESHOLD:
            cv2.putText(frame, "RIGHT EYE BLINK", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mouthAR > MOUTH_AR_THRESHOLD:
            cv2.putText(frame, "MOUTH OPEN", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Landmarks", frame)
    key = cv2.waitKey(1)
    if key == 27:           # ESC key
        break

video.release()
cv2.destroyAllWindows()

