import cv2
import PySimpleGUI as sg
import numpy as np
from pygrabber.dshow_graph import FilterGraph

def main():
    # Get the list of available webcams
    webcam_list = get_available_cameras()

    # GUI layout
    layout = [
        [sg.Text("Select a webcam: "), sg.Combo(webcam_list, key="-WEBCAM-", enable_events=True)],
        [sg.Button("Start", key="-START-", disabled=True), sg.Button("Exit")]
    ]

    # Create the window
    window = sg.Window("Webcam Selector", layout, finalize=True)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
        elif event == "-WEBCAM-":
            window["-START-"].update(disabled=values["-WEBCAM-"] == "")
        elif event == "-START-":
            selected_webcam = values["-WEBCAM-"]
            if selected_webcam:
                capture_and_display(selected_webcam[0])

    window.close()


def get_available_cameras() :

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

def capture_and_display(webcam_id):
    cap = cv2.VideoCapture(int(webcam_id.split()[-1]), cv2.CAP_DSHOW)

    if not cap.isOpened():
        sg.popup_error("Error opening the selected webcam.")
        return

    sg.popup(f"Selected webcam: {webcam_id}", "Click OK to start capturing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow("Webcam Display", frame)

        # Check for user exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
