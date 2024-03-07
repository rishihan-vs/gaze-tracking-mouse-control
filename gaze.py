import time
import cv2
import numpy as np
import random
import dlib
import sys
import pyautogui
import mouse

def black_screen(webcam_size):
    page = (np.zeros((int(webcam_size[0]), int(webcam_size[1]), 3))).astype('uint8')
    return page

def is_blinking(eye_coordinates):
    blinking = False

    major_axis = np.sqrt((eye_coordinates[1][0]-eye_coordinates[0][0])**2 + (eye_coordinates[1][1]-eye_coordinates[0][1])**2)
    minor_axis = np.sqrt((eye_coordinates[3][0]-eye_coordinates[2][0])**2 + (eye_coordinates[3][1]-eye_coordinates[2][1])**2)

    ratio = minor_axis / major_axis

    if ratio < 0.2:
        blinking = True

    return blinking

camera = cv2.VideoCapture(0)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

(a,b) = pyautogui.size()
screen_size = (b,a)
print(screen_size)

webcam_size = (int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
print(webcam_size)

coord_offset = (50, 50)

calibration_screen = black_screen(webcam_size)
screen = black_screen(screen_size)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

corners = [(coord_offset),
           (webcam_size[1] - coord_offset[1], webcam_size[0] - coord_offset[0]),
           (webcam_size[1] - coord_offset[1], coord_offset[1]),
           (coord_offset[0], webcam_size[0] - coord_offset[0])]
print(corners)
calibration_cut = []
corner = 0

while(corner < 4): # calibration of 4 corners

    ret, frame = camera.read()   # Capture frame
    frame = cv2.flip(frame, 1)  # rotate / flip

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

    # messages for calibration
    cv2.putText(calibration_screen, 'calibration: look at', tuple((np.array(webcam_size)/7).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    cv2.putText(calibration_screen, 'the circle and blink', tuple((np.array(webcam_size)/7 + 50).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    cv2.circle(calibration_screen, corners[corner], 40, (0, 255, 0), -1)

    # detect faces in frame
    faces = detector(gray_scale_frame)
    if len(faces)> 1:
        print('please avoid multiple faces.')
        sys.exit()

    for face in faces:
        # display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

        landmarks = predictor(gray_scale_frame, face) # find points in face
        for point in range(0, 68):
            x = landmarks.part(point).x
            y = landmarks.part(point).y
            cv2.circle(frame, (x, y), 4, (0,0,255), 2)

        # get position of right eye and display lines
        x_left = (landmarks.part(42).x, landmarks.part(42).y)
        x_right = (landmarks.part(45).x, landmarks.part(45).y)
        y_top = (int((landmarks.part(43).x + landmarks.part(44).x)/2), int((landmarks.part(43).y + landmarks.part(44).y)/2))
        y_bottom = (int((landmarks.part(47).x + landmarks.part(46).x)/2), int((landmarks.part(47).y + landmarks.part(46).y)/2))

        right_eye_coordinates = (x_left, x_right, y_top, y_bottom)
        cv2.line(frame, right_eye_coordinates[0], right_eye_coordinates[1], (0,255,0), 2)
        cv2.line(frame, right_eye_coordinates[2], right_eye_coordinates[3], (0,255,0), 2)

    # define the coordinates of the pupil from the centroid of the right eye
    pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')

    if is_blinking(right_eye_coordinates):
        calibration_cut.append(pupil_coordinates)

        # visualize message
        cv2.putText(calibration_screen, 'ok',
                    tuple(np.array(corners[corner])-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 5)
        # to avoid is_blinking=True in the next frame
        time.sleep(0.3)
        corner = corner + 1

    print(calibration_cut, '    len: ', len(calibration_cut))

    # cv2.namedWindow('projection', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('projection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow('projection')
    cv2.imshow('projection',calibration_screen)
    cv2.moveWindow('projection', (screen_size[1] - webcam_size[1])//2, (screen_size[0] - webcam_size[0])//2)

    cv2.namedWindow('frame')
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

cv2.destroyAllWindows()



x_cut_max = np.transpose(np.array(calibration_cut))[0].max()
x_cut_min = np.transpose(np.array(calibration_cut))[0].min()
y_cut_max = np.transpose(np.array(calibration_cut))[1].max()
y_cut_min = np.transpose(np.array(calibration_cut))[1].min()

offset_calibrated_cut = [ x_cut_min, y_cut_min ]

# cv2.putText(calibration_screen, 'calibration done. please wait',
#             tuple((np.array(webcam_size)/3).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.4,(255, 255, 255), 3)
# cv2.namedWindow('projection')
# cv2.imshow('projection',calibration_screen)
# time.sleep(2)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()



while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    cut_frame = np.copy(frame[y_cut_min:y_cut_max, x_cut_min:x_cut_max, :])

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_scale_frame)  # detect faces in frame
    if len(faces)> 1:
        print('please avoid multiple faces..')
        sys.exit()

    for face in faces:
        landmarks = predictor(gray_scale_frame, face)
        for point in range(0, 68):
            x = landmarks.part(point).x
            y = landmarks.part(point).y
            cv2.circle(frame, (x, y), 4, (0,0,255), 2)

        # get position of right eye and display lines
        x_left = (landmarks.part(42).x, landmarks.part(42).y)
        x_right = (landmarks.part(45).x, landmarks.part(45).y)
        y_top = (int((landmarks.part(43).x + landmarks.part(44).x)/2), int((landmarks.part(43).y + landmarks.part(44).y)/2))
        y_bottom = (int((landmarks.part(47).x + landmarks.part(46).x)/2), int((landmarks.part(47).y + landmarks.part(46).y)/2))

        right_eye_coordinates = (x_left, x_right, y_top, y_bottom)
        cv2.line(frame, right_eye_coordinates[0], right_eye_coordinates[1], (0,255,0), 2)
        cv2.line(frame, right_eye_coordinates[2], right_eye_coordinates[3], (0,255,0), 2)

    # define the coordinates of the pupil from the centroid of the right eye
    pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')

    # work on the calibrated cut-frame
    pupil_on_cut = np.array([pupil_coordinates[0] - offset_calibrated_cut[0], pupil_coordinates[1] - offset_calibrated_cut[1]])
    eye_radius = int(np.sqrt((right_eye_coordinates[3][0]-right_eye_coordinates[2][0])**2 + (right_eye_coordinates[3][1]-right_eye_coordinates[2][1])**2))
    
    cv2.circle(cut_frame, (pupil_on_cut[0], pupil_on_cut[1]), int(eye_radius/1.5), (255, 0, 0), 3)

    in_frame_cut = False
    condition_x = ((pupil_on_cut[0] > 0) & (pupil_on_cut[0] < cut_frame.shape[1]))
    condition_y = ((pupil_on_cut[1] > 0) & (pupil_on_cut[1] < cut_frame.shape[0]))
    if condition_x and condition_y:
        in_frame_cut = True

    scale = (np.array(screen[:,:, 0].shape) / np.array(cut_frame[:,:, 0].shape))
    projected_point = (pupil_on_cut * scale).astype('int')

    if in_frame_cut:
        pupil_on_screen = projected_point

        # print("PUPIL: ", pupil_on_screen)

        # cv2.circle(screen, (pupil_on_screen[0], pupil_on_screen[1]), 40, (0, 255, 0), 3)

        # pyautogui.moveTo(pupil_on_screen[0], pupil_on_screen[1])
        # pyautogui.FAILSAFE = False

        mouse.move(pupil_on_screen[0], pupil_on_screen[1])

        # print("MOUSE: ", pyautogui.position())


    # cv2.namedWindow('projection')
    # cv2.imshow('projection', screen)

    cv2.namedWindow('frame')
    cv2.imshow('frame', cv2.resize(frame, (int(frame.shape[1] *0.3), int(frame.shape[0] *0.3))))
    
    cv2.namedWindow('cut_frame')
    cv2.imshow('cut_frame', cv2.resize(cut_frame, (int(cut_frame.shape[1] *4.5), int(cut_frame.shape[0] *4.5))))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()