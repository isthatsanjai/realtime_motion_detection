import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Initialize variables for motion detection
first_frame = None
motion_detected = False
motion_counter = 0
no_motion_counter = 0
recording = False
recording_file = 'motion_recording.avi'

# Function to detect motion
def detect_motion(frame, first_frame, threshold=25):
    # Convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If first frame is None, initialize it
    if first_frame is None:
        return gray, False

    # Compute the absolute difference between the current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any significant contour is found
    motion_detected = any(cv2.contourArea(contour) > 500 for contour in contours)

    return gray, motion_detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray, motion = detect_motion(frame, first_frame)

    # If first_frame is None, initialize it
    if first_frame is None:
        first_frame = gray
        continue

    if motion:
        motion_counter += 1
        no_motion_counter = 0
    else:
        no_motion_counter += 1
        motion_counter = 0

    # Start recording if motion is detected
    if motion_counter >= 5 and not recording:
        recording = True
        out = cv2.VideoWriter(recording_file, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print("Started recording.")

    # Stop recording if no motion is detected for a while
    if no_motion_counter >= 50 and recording:
        recording = False
        out.release()
        out = None
        print("Stopped recording.")

    # Write the frame if recording
    if recording:
        out.write(frame)

    # Show the current frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

