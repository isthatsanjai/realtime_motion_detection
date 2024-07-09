import cv2
import os
from datetime import datetime

# Initialize motion detection variables
motion_counter = 0
no_motion_counter = 0
recording = False

# Define the directory where recordings will be saved
output_directory = 'recordings'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Create Background Subtractor object
backSub = cv2.createBackgroundSubtractorMOG2()

# Initialize video capture
vcap = cv2.VideoCapture(0)

# Define frame size and codec for output video
frameWidth = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameSize = (frameWidth, frameHeight)


# Define codec for recording motion
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

while True:
    ret, frame = vcap.read()
    if not ret:
        break

    # Apply bilateral filter to the frame
    filtered_frame = cv2.bilateralFilter(frame, d=15, sigmaColor=75, sigmaSpace=75)

    # Update the background model and get foreground mask
    fgMask = backSub.apply(filtered_frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion = False

    # Process each contour
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue

        motion = True

        # Draw bounding rectangle and label
        x, y, w, h = cv2.boundingRect(c)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Update motion counters
    if motion:
        motion_counter += 1
        no_motion_counter = 0
    else:
        no_motion_counter += 1
        motion_counter = 0

    # Start recording if motion is detected
    if motion_counter >= 25 and not recording:
        recording = True
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        recording_file = os.path.join(output_directory, f'motion_recording_{dt_string}.avi')
        out = cv2.VideoWriter(recording_file, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print(f"Started recording: {recording_file}")

    # Stop recording if no motion is detected for a while
    if no_motion_counter >= 20 and recording:
        recording = False
        out.release()
        out = None
        print("Stopped recording.")

    # Write the frame if recording
    if recording:
        cv2.putText(frame, "Started Recording", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Add text overlay
        out.write(frame)

    # Display the frame and foreground mask
    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
if out is not None:
    out.release()
vcap.release()
cv2.destroyAllWindows()
