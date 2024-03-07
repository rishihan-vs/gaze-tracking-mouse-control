import cv2
import PySimpleGUI as sg
import numpy as np
from pygrabber.dshow_graph import FilterGraph
import time
import random
import dlib
import sys
import pyautogui
import mouse
from imutils import face_utils
from scipy.spatial import distance as dist
from OneEuroFilter import OneEuroFilter
from typing import Tuple

def get_available_cameras():

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    webcam_list = []
    for index, name in available_cameras.items():  # Check up to 10 webcams
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            webcam_list.append(f"{index}: {name}")
            cap.release()
    return webcam_list

def start_window():
    # Get the list of available webcams
    webcam_list = get_available_cameras()

    mode_list = {
        "Normal mode" : "Normal mode instructions: \nMouth open/smile for less than 2 seconds - LEFT CLICK \nMouth open/smile for 2 or more seconds - RIGHT CLICK \nEye blinking for 2 seconds - Toggle scrolling mode",
        "Speech mode" : "Speech mode instructions: \nSay \"Left click\" - LEFT CLICK \nSay \"Right click\" - RIGHT CLICK \nSay \"Scrolling mode\" - Toggle scrolling mode"
        }

    # GUI layout
    layout = [
        [sg.Column([
            [sg.Text("Welcome to the Eye Mouse application", size=(60, 1), font=("Helvetica", 20, "bold"), justification="center")],
            [sg.Text("When using this tool, please make sure your face is very well lit, and there are no lights behind you.", size=(60, 2), font=("Helvetica", 15), justification="center")],
            [sg.Text("Select a webcam: ", font=("Helvetica", 15), justification="center"), sg.Combo(webcam_list, key="-WEBCAM-", enable_events=True, readonly=True)],
            [sg.Text("Select mode: ", font=("Helvetica", 15), justification="center"), sg.Combo(list(mode_list.keys()), key="-MODE-", enable_events=True, readonly=True)],
            [sg.Text("", font=("Helvetica", 15), text_color="orange", key="-INS-")],
            [sg.Button("Start", key="-START-", disabled=True), sg.Button("Exit")]
        ], element_justification="center")]
        
    ]

    # Create the window
    window = sg.Window("Eye Mouse", layout, finalize=True, keep_on_top=True)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
        elif event == "-WEBCAM-":
            window["-START-"].update(disabled=values["-WEBCAM-"] == "")
        elif event == "-MODE-":
            window["-START-"].update(disabled=values["-MODE-"] == "")
            selected_option = values["-MODE-"]
            displayed_text = mode_list.get(selected_option, "")
            window["-INS-"].update(displayed_text)
            
        elif event == "-START-":
            selected_webcam = values["-WEBCAM-"]
            selected_mode = values["-MODE-"]
            if selected_webcam and selected_mode:
                window.close()
                return selected_webcam[0], selected_mode

    window.close()

def black_screen(webcam_size):
    page = (np.zeros((int(webcam_size[0]), int(webcam_size[1]), 3))).astype('uint8')
    return page

def is_blinking(eye_coordinates):
    blinking = False

    major_axis = np.sqrt((eye_coordinates[1][0]-eye_coordinates[0][0])**2 + (eye_coordinates[1][1]-eye_coordinates[0][1])**2)
    minor_axis = np.sqrt((eye_coordinates[3][0]-eye_coordinates[2][0])**2 + (eye_coordinates[3][1]-eye_coordinates[2][1])**2)

    ratio = minor_axis / major_axis

    if ratio < 0.22:
        blinking = True

    return blinking

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


EYE_AR_THRESHOLD = 0.2
MOUTH_AR_THRESHOLD = 0.28

webcam_id, mode = start_window()
webcam = int(webcam_id.split()[-1])

print(webcam_id, mode)

camera = cv2.VideoCapture(webcam)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

(a,b) = pyautogui.size()            # Get monitor resolution
screen_size = (b,a)
print(screen_size)

webcam_size = (int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))           # Get webcam resolution
print(webcam_size)

coord_offset = (50, 50)

screen = black_screen(screen_size)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

corners = [(coord_offset),
           (webcam_size[1] - coord_offset[1], webcam_size[0] - coord_offset[0]),
           (webcam_size[1] - coord_offset[1], coord_offset[1]),
           (coord_offset[0], webcam_size[0] - coord_offset[0])]
print(corners)

# -------------------------- CALIBRATION ------------------------------------
calibration_success = False

while (not calibration_success):
    calibration_cut = []
    corner = 0
    calibration_screen = black_screen(webcam_size)

    while(corner < 4): # calibration of 4 corners

        ret, frame = camera.read()   # Capture frame
        frame = cv2.flip(frame, 1)  # rotate / flip

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

        # messages for calibration
        cv2.putText(calibration_screen, 'Calibration: Look at the top left, bottom right, top right,', (int(webcam_size[0]/7), int(webcam_size[1]/7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        cv2.putText(calibration_screen, 'left, bottom right, top right, and bottom left of your', (int(webcam_size[0]/7), int(webcam_size[1]/7 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        cv2.putText(calibration_screen, 'screen (in that order) and blink for each corner', (int(webcam_size[0]/7), int(webcam_size[1]/7 + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        # cv2.circle(calibration_screen, corners[corner], 40, (0, 255, 0), -1)

        # detect faces in frame
        faces = detector(gray_scale_frame)
        # if len(faces)> 1:
        #     print('please avoid multiple faces.')
        #     sys.exit()

        for face in faces:
            face = faces[0]

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

        # Define the coordinates of the pupil from the centre of the right eye
        pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0)

        if is_blinking(right_eye_coordinates):
            calibration_cut.append(pupil_coordinates)

            cv2.putText(calibration_screen, 'OK',
                        (corners[corner][0]-50, corners[corner][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 5)
            
            time.sleep(0.5)                             # To avoid is_blinking = True in the next frame
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

    cut_frame = np.copy(frame[int(y_cut_min):int(y_cut_max), int(x_cut_min):int(x_cut_max), :])
    cond = cut_frame.shape[1]/cut_frame.shape[0]                # Check cut frame aspect ratio is good enough for tracking
    print(cond)

    if 1.5 < cond < 2.2:
        calibration_success = True

        calibration_screen = black_screen(webcam_size)
        cv2.putText(calibration_screen, 'Calibration done. Please wait',
                    tuple((np.array(webcam_size)/3).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.4,(255, 255, 255), 3)
        cv2.namedWindow('projection')
        cv2.imshow('projection',calibration_screen)
        time.sleep(2)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        sg.popup("Calibration error: Please perform calibration again")
        # sg.popup_auto_close("Calibration error: Please perform calibration again", auto_close=True, auto_close_duration=2)


# --------------------------------------------------------------------------------

start_time_mouth = None
start_time_eye = None
scrolling_mode = False

x_queue = []
y_queue = []

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    cut_frame = np.copy(frame[int(y_cut_min):int(y_cut_max), int(x_cut_min):int(x_cut_max), :])

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_scale_frame)  # detect faces in frame
    # if len(faces)> 1:
    #     print('please avoid multiple faces..')
    #     sys.exit()
    

    for face in faces:
        face = faces[0]     # Choose first face that was detected - ignore background faces that may come in frame
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

        shape = face_utils.shape_to_np(landmarks)

        # frame is flipped so eyes need to be swapped
        rightEye = shape[lStart:lEnd]
        leftEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        mouth = shape[mStart:mEnd]
        mouthAR = mouth_aspect_ratio(mouth)

        if mouthAR > MOUTH_AR_THRESHOLD and start_time_mouth is None:
            start_time_mouth = time.time()
        elif not mouthAR > MOUTH_AR_THRESHOLD and start_time_mouth is not None:
            duration = time.time() - start_time_mouth
            if duration < 2:                                # if mouth open for less than 2 seconds, then perform left click
                mouse.click("left")
            else:                                           # if mouth open for 2 or more seconds, turn on scrolling mode
                mouse.click("right")
            start_time_mouth = None

        if is_blinking(right_eye_coordinates) and start_time_eye is None:
            start_time_eye = time.time()
        elif not is_blinking(right_eye_coordinates) and start_time_eye is not None:
            duration = time.time() - start_time_eye
            if duration >= 2:
                scrolling_mode = not scrolling_mode
                print("Scrolling mode")
            start_time_eye = None

    # define the coordinates of the pupil from the centroid of the right eye
    pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0)

    # work on the calibrated cut-frame
    pupil_on_cut = np.array([pupil_coordinates[0] - offset_calibrated_cut[0], pupil_coordinates[1] - offset_calibrated_cut[1]])
    eye_radius = np.sqrt((right_eye_coordinates[3][0]-right_eye_coordinates[2][0])**2 + (right_eye_coordinates[3][1]-right_eye_coordinates[2][1])**2)
    
    cv2.circle(cut_frame, (int(pupil_on_cut[0]), int(pupil_on_cut[1])), int(eye_radius/1.5), (255, 0, 0), 3)

    in_frame_cut = False
    condition_x = ((pupil_on_cut[0] > 0) & (pupil_on_cut[0] < cut_frame.shape[1]))
    condition_y = ((pupil_on_cut[1] > 0) & (pupil_on_cut[1] < cut_frame.shape[0]))
    if condition_x and condition_y:
        in_frame_cut = True

    scale = (np.array(screen[:,:, 0].shape) / np.array(cut_frame[:,:, 0].shape))
    projected_point = (pupil_on_cut * (scale+3))                # Adding 3 to scale so eye doesn't have to be on edge of cut frame to reach screen edges

    if in_frame_cut:
        pupil_on_screen = projected_point

        # print("PUPIL: ", pupil_on_screen)

        # cv2.circle(screen, (pupil_on_screen[0], pupil_on_screen[1]), 40, (0, 255, 0), 3)

        # pyautogui.moveTo(pupil_on_screen[0], pupil_on_screen[1])
        # pyautogui.FAILSAFE = False

        if len(x_queue) > 3:
            x_queue.pop(0)

        if len(y_queue) > 3:
            y_queue.pop(0)

        x_queue.append(pupil_on_screen[0])
        y_queue.append(pupil_on_screen[1])

        avg_x = sum(x_queue) / len(x_queue)
        avg_y = sum(y_queue) / len(y_queue)

        # filtered_x = f(pupil_on_screen[0])
        # filtered_y = f(pupil_on_screen[1])
        # mouse.move(filtered_x, filtered_y)

        # mouse.move(pupil_on_screen[0], pupil_on_screen[1])
        if not scrolling_mode:
            mouse.move(avg_x, avg_y)
        else:
            dy = pupil_on_cut[1] - (cut_frame.shape[0] // 2)                # distance in y-direction of pupil in cut frame from centre of cut frame window

            if -2 < dy < 2:
                dy = 0

            mouse.wheel(-dy*0.6)

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





# https://mouseaccuracy.com/
# for testing