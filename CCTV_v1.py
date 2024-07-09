import cv2


motion_detected = False
motion_counter = 0
no_motion_counter = 0
recording = False
recording_file = 'motion_recording.avi'


# Create Background Subtractor object
backSub = cv2.createBackgroundSubtractorMOG2()

# Initialize video capture
vcap = cv2.VideoCapture(0)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Define frame size and codec for output video
frameWidth = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameSize = (frameWidth, frameHeight)
codec = cv2.VideoWriter_fourcc(*'mp4v')
outputVideo = cv2.VideoWriter("motion_detected.mp4", codec, vcap.get(cv2.CAP_PROP_FPS), frameSize, True)

while True:
    ret, frame = vcap.read()
    if not ret:
        break

    # Apply bilateral filter to the frame
    filtered_frame = cv2.bilateralFilter(frame, d=25, sigmaColor=75, sigmaSpace=75)  # Adjust parameters as needed

    # Update the background model and get foreground mask
    fgMask = backSub.apply(filtered_frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for c in contours:
        motion = False
        if cv2.contourArea(c) < 500:
            continue
            motion = True

        # Draw bounding rectangle and label
        x, y, w, h = cv2.boundingRect(c)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Write the frame to the output video
    outputVideo.write(frame)

    # Display the frame and foreground mask
    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)
    if motion:
        motion_counter += 1
        no_motion_counter = 0
    else:
        no_motion_counter += 1
        motion_counter = 0

        # Start recording if motion is detected
    if motion_counter >= 200 and not recording:
        recording = True
        out = cv2.VideoWriter(recording_file, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print("Started recording.")

        # Stop recording if no motion is detected for a while
    if no_motion_counter >= 50 and recording:
        recording = False
        out.release()
        out = None
        print("Stopped recording.")
        break
        # Write the frame if recording
    if recording:
        out.write(frame)

        # Show the current frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
outputVideo.release()
vcap.release()
cv2.destroyAllWindows()