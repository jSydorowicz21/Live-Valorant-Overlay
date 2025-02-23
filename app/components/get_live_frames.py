import os
import time

import cv2
import numpy as np
import socketio
import win32gui

import dxcam  # Requires: pip install dxcam

from live_details import LiveDetails

live_details_helper = LiveDetails()

# standard Python Socket.IO client
socket_io = socketio.Client()
print("client is created")

def after_connect(args):
    print('after connect', args['data'])

def kill_self(args):
    print('kill_self', args['data'])
    os._exit(0)

socket_io.on('after connect', after_connect)
socket_io.on('kill_self', kill_self)

def get_hwnd():
    window_name = "VALORANT  "
    hwnd = win32gui.FindWindow(None, window_name)
    print(f"Looking for window: '{window_name}', found hwnd: {hwnd}")
    return hwnd

def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError

# Global index for debugging/saving frames
frame_index = 0

def start_frame_grabbing():
    global frame_index
    hwnd = get_hwnd()
    if hwnd == 0:
        print("Valorant window not found")
        return

    # Get the window's coordinates
    rect = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = rect
    w = right - left
    h = bottom - top
    print(f"Capturing region: left: {left}, top: {top}, width: {w}, height: {h}")

    # Create and start the dxcam capture using the Desktop Duplication API.
    # output_color "BGR" is set so that the frames work directly with OpenCV.
    cam = dxcam.create(output_color="BGR")
    cam.start(region=(left, top, w, h), target_fps=2)

    # Poll continuously for new frames.
    while True:
        frame = cam.get_latest_frame()
        if frame is None:
            continue

        # Optionally, save frame for debugging as a BMP image.
        bmp_filename = f"out{frame_index}.bmp"
        cv2.imwrite(bmp_filename, frame)
        frame_index += 1

        # Process the frame for live details
        detected_events = live_details_helper.get_live_details(frame)
        if detected_events:
            socket_io.emit('new_event', {'event': detected_events})

if __name__ == "__main__":
    socket_io.connect('http://127.0.0.1:4445/')
    # Start frame grabbing in a background task.
    socket_io.start_background_task(start_frame_grabbing)
    socket_io.wait()
