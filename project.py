import cv2
import PySimpleGUI as sg
import numpy as np
from pygrabber.dshow_graph import FilterGraph
import time
import dlib
import sys
import pyautogui
import mouse
from imutils import face_utils
from scipy.spatial import distance as dist

# Returns list of webcams found on device
def get_available_cameras():

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    webcam_list = []
    for index, name in available_cameras.items():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            webcam_list.append(f"{index}: {name}")
            cap.release()
    return webcam_list

# Creates GUI for start window
def start_window():
    # Get the list of available webcams
    webcam_list = get_available_cameras()
    
    eye_list = ["Left", "Right"]

    # GUI layout
    layout = [
        [sg.Column([
            [sg.Text("Welcome to the Eye Mouse application", size=(60, 1), font=("Helvetica", 20, "bold"), justification="center", background_color="#003997")],
            [sg.Text("When using this tool, please make sure your face is very well lit, and there are no lights behind you.", size=(60, 2), font=("Helvetica", 15), justification="center", background_color="#003997")],
            [sg.Text("Select a webcam: ", font=("Helvetica", 15), justification="center", background_color="#003997"), sg.Combo(webcam_list, key="-WEBCAM-", enable_events=True, readonly=True)],
            [sg.Text("Select dominant eye: ", font=("Helvetica", 15), justification="center", background_color="#003997"), sg.Combo(eye_list, key="-EYE-", enable_events=True, readonly=True)],
            [sg.Text("Instructions: \nMouth open/smile for less than 2 seconds - LEFT CLICK \nMouth open/smile for 2 or more seconds - RIGHT CLICK \nEye blinking for 2 seconds - Toggle scrolling mode", font=("Helvetica", 15), text_color="orange", key="-INS-", background_color="#003997")],
            [sg.Button("Start", key="-START-", disabled=True), sg.Button("Exit")]
        ], element_justification="center", background_color="#003997")]
        
    ]

    # Create the window
    window = sg.Window("Eye Mouse", layout, finalize=True, keep_on_top=True, background_color="#003997")

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            sys.exit()
        elif event == "-WEBCAM-":
            window["-START-"].update(disabled=values["-WEBCAM-"] == "")
        elif event == "-EYE-":
            window["-START-"].update(disabled=values["-EYE-"] == "")
            
        elif event == "-START-":
            selected_webcam = values["-WEBCAM-"]
            selected_eye = values["-EYE-"]
            if selected_webcam and selected_eye:
                window.close()
                return selected_webcam[0], selected_eye

# Creates black page for calibration and screen area
def black_screen(webcam_size):
    page = (np.zeros((int(webcam_size[0]), int(webcam_size[1]), 3))).astype('uint8')
    return page

# Detects if eyes are blinking
def is_blinking(eye_coordinates):
    blinking = False

    major_axis = np.sqrt((eye_coordinates[1][0]-eye_coordinates[0][0])**2 + (eye_coordinates[1][1]-eye_coordinates[0][1])**2)
    minor_axis = np.sqrt((eye_coordinates[3][0]-eye_coordinates[2][0])**2 + (eye_coordinates[3][1]-eye_coordinates[2][1])**2)

    ratio = minor_axis / major_axis

    if ratio < 0.22:
        blinking = True

    return blinking

# Detects if mouth is opened/closed
def mouth_aspect_ratio(mouth):
     A = dist.euclidean(mouth[2], mouth[7])
     B = dist.euclidean(mouth[3], mouth[5])

     C = dist.euclidean(mouth[0], mouth[4])

     mouthAR = (A + B) / (2.0 * C)

     return mouthAR

# Returns eye coordinates for chosen eye
def get_eye_position(eye_choice):

    if eye_choice == "Right":
        # get position of right eye and display lines
        x_left = (landmarks.part(42).x, landmarks.part(42).y)
        x_right = (landmarks.part(45).x, landmarks.part(45).y)
        y_top = (int((landmarks.part(43).x + landmarks.part(44).x)/2), int((landmarks.part(43).y + landmarks.part(44).y)/2))
        y_bottom = (int((landmarks.part(47).x + landmarks.part(46).x)/2), int((landmarks.part(47).y + landmarks.part(46).y)/2))

    else:
        # get position of left eye and display lines
        x_left = (landmarks.part(36).x, landmarks.part(36).y)
        x_right = (landmarks.part(39).x, landmarks.part(39).y)
        y_top = (int((landmarks.part(37).x + landmarks.part(38).x)/2), int((landmarks.part(37).y + landmarks.part(38).y)/2))
        y_bottom = (int((landmarks.part(41).x + landmarks.part(40).x)/2), int((landmarks.part(41).y + landmarks.part(40).y)/2))

    return x_left, x_right, y_top, y_bottom

# Threshold for detecting whether mouth is closed/opened
MOUTH_AR_THRESHOLD = 0.28

webcam_id, eye_choice = start_window()
webcam = int(webcam_id.split()[-1])

print(webcam_id, eye_choice)

camera = cv2.VideoCapture(webcam)

(a,b) = pyautogui.size()            # Get monitor resolution
screen_size = (b,a)
print(screen_size)

webcam_size = (int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))           # Get webcam resolution
print(webcam_size)

screen = black_screen(screen_size)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

corners = [(50, 50),
           (webcam_size[1] - 50, webcam_size[0] - 50),
           (webcam_size[1] - 50, 50),
           (50, webcam_size[0] - 50)]
print(corners)

# -------------------------- CALIBRATION ------------------------------------
calibration_success = False

while (not calibration_success):
    calibration_cut = []
    corner = 0
    calibration_screen = black_screen(webcam_size)

    while(corner < 4): # calibration of 4 corners

        _, frame = camera.read()   # Capture frame
        frame = cv2.flip(frame, 1)  # Flip frame

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.putText(calibration_screen, 'Calibration: Look at the top left, bottom right,', (int(webcam_size[0]/7), int(webcam_size[1]/7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
        cv2.putText(calibration_screen, 'top right, and bottom left of your screen', (int(webcam_size[0]/7), int(webcam_size[1]/7 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
        cv2.putText(calibration_screen, '(in that order) and blink for each corner', (int(webcam_size[0]/7), int(webcam_size[1]/7 + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)

        faces = detector(gray_scale_frame)                  # Detect faces in frame

        for face in faces:
            face = faces[0]                                 # Choose first face that was detected - ignore background faces that may come in frame

            landmarks = predictor(gray_scale_frame, face)               # Find points on face
            for point in range(0, 68):
                x = landmarks.part(point).x
                y = landmarks.part(point).y
                cv2.circle(frame, (x, y), 2, (0,0,255), 4)
                
            if eye_choice == "Right":
                (x_left, x_right, y_top, y_bottom) = get_eye_position("Right")
            else:
                (x_left, x_right, y_top, y_bottom) = get_eye_position("Left")

            eye_coordinates = (x_left, x_right, y_top, y_bottom)
            cv2.line(frame, eye_coordinates[0], eye_coordinates[1], (0,255,0), 2)
            cv2.line(frame, eye_coordinates[2], eye_coordinates[3], (0,255,0), 2)

        # Define coordinates of pupil from the centre of the eye
        pupil_coordinates = np.mean([eye_coordinates[2], eye_coordinates[3]], axis = 0)

        if is_blinking(eye_coordinates):
            calibration_cut.append(pupil_coordinates)

            cv2.putText(calibration_screen, 'OK', (corners[corner][0]-50, corners[corner][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 5)
            
            time.sleep(0.5)                             # To avoid is_blinking = True in the next frame
            corner += 1

        # print(calibration_cut)

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

    cut_frame = np.copy(frame[int(y_cut_min):int(y_cut_max), int(x_cut_min):int(x_cut_max), :])
    cond = cut_frame.shape[1]/cut_frame.shape[0]                # Check cut frame aspect ratio is good enough for tracking
    print(cond)

    if 1.5 < cond < 2.2:
        calibration_success = True

    else:
        sg.popup("Calibration error: Please perform calibration again")

# --------------------------------------------------------------------------------

start_time_mouth = None
start_time_eye = None
scrolling_mode = False

x_queue = []
y_queue = []

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)

    cut_frame = np.copy(frame[int(y_cut_min):int(y_cut_max), int(x_cut_min):int(x_cut_max), :])

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_scale_frame)                  # Detect faces in frame

    for face in faces:
        face = faces[0]                                 # Choose first face that was detected - ignore background faces that may come in frame
        landmarks = predictor(gray_scale_frame, face)
        for point in range(0, 68):
            x = landmarks.part(point).x
            y = landmarks.part(point).y
            cv2.circle(frame, (x, y), 2, (0,0,255), 4)

        if eye_choice == "Right":
            (x_left, x_right, y_top, y_bottom) = get_eye_position("Right")
        else:
            (x_left, x_right, y_top, y_bottom) = get_eye_position("Left")

        eye_coordinates = (x_left, x_right, y_top, y_bottom)
        cv2.line(frame, eye_coordinates[0], eye_coordinates[1], (0,255,0), 2)
        cv2.line(frame, eye_coordinates[2], eye_coordinates[3], (0,255,0), 2)

        shape = face_utils.shape_to_np(landmarks)

        mouth = shape[mStart:mEnd]
        mouthAR = mouth_aspect_ratio(mouth)

        if mouthAR > MOUTH_AR_THRESHOLD and start_time_mouth is None:
            start_time_mouth = time.time()
        elif not mouthAR > MOUTH_AR_THRESHOLD and start_time_mouth is not None:
            duration = time.time() - start_time_mouth
            if duration < 2:                                # if mouth open for less than 2 seconds, then perform left click
                mouse.click("left")
                cv2.putText(frame, "LEFT CLICK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:                                           # if mouth open for 2 or more seconds, then perform right click
                mouse.click("right")
                cv2.putText(frame, "RIGHT CLICK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            start_time_mouth = None

        if is_blinking(eye_coordinates) and start_time_eye is None:
            start_time_eye = time.time()
        elif not is_blinking(eye_coordinates) and start_time_eye is not None:
            duration = time.time() - start_time_eye
            if duration >= 2:                               # if eye is closed for 2 or more seconds, turn on scrolling mode
                scrolling_mode = not scrolling_mode
                print("Scrolling mode")
            start_time_eye = None

    # Define coordinates of pupil from the centre of the eye
    pupil_coordinates = np.mean([eye_coordinates[2], eye_coordinates[3]], axis = 0)

    # Work on cut frame
    pupil_in_cut_frame = np.array([pupil_coordinates[0] - x_cut_min, pupil_coordinates[1] - y_cut_min])

    eye_radius = np.sqrt((eye_coordinates[3][0]-eye_coordinates[2][0])**2 + (eye_coordinates[3][1]-eye_coordinates[2][1])**2)
    cv2.circle(cut_frame, (int(pupil_in_cut_frame[0]), int(pupil_in_cut_frame[1])), int(eye_radius/1.5), (0, 0, 255), 3)

    # Checks if pupil is in cut frame
    in_cut_frame = False
    condition_x = ((pupil_in_cut_frame[0] > 0) & (pupil_in_cut_frame[0] < cut_frame.shape[1]))
    condition_y = ((pupil_in_cut_frame[1] > 0) & (pupil_in_cut_frame[1] < cut_frame.shape[0]))
    if condition_x and condition_y:
        in_cut_frame = True

    scale = (np.array(screen[:,:, 0].shape) / np.array(cut_frame[:,:, 0].shape))
    projected_point = (pupil_in_cut_frame * (scale+3))                # Adding 3 to scale so eye doesn't have to be on edge of cut frame to reach screen edges

    # If pupil in cut frame, then move mouse if scrolling mode is off, or scroll if scrolling mode is on
    if in_cut_frame:
        mouse_coordinates = projected_point

        # Use queues for moving average filter
        if len(x_queue) > 3:
            x_queue.pop(0)

        if len(y_queue) > 3:
            y_queue.pop(0)

        x_queue.append(mouse_coordinates[0])
        y_queue.append(mouse_coordinates[1])

        avg_x = sum(x_queue) / len(x_queue)
        avg_y = sum(y_queue) / len(y_queue)

        if not scrolling_mode:
            mouse.move(avg_x, avg_y)
        else:
            cv2.putText(frame, "SCROLLING MODE", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            dy = pupil_in_cut_frame[1] - (cut_frame.shape[0] // 2)                # distance in y-direction of pupil in cut frame from centre of cut frame window

            # Dead zone at centre of cut frame - no accidental scrolling when trying not to scroll
            if -3 < dy < 3:
                dy = 0

            mouse.wheel(-dy*0.6)

    cv2.namedWindow('frame')
    cv2.imshow('frame', cv2.resize(frame, (int(frame.shape[1] *0.3), int(frame.shape[0] *0.3))))
    
    cv2.namedWindow('cut_frame')
    cv2.imshow('cut_frame', cv2.resize(cut_frame, (int(cut_frame.shape[1] *4.5), int(cut_frame.shape[0] *4.5))))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

