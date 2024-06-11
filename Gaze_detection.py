import cv2
import dlib
import numpy as np
from IPython.display import display, clear_output
import PIL.Image
from io import BytesIO
import time

def detect_gaze(landmarks, frame, gray):
    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
    right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                 (landmarks.part(43).x, landmarks.part(43).y),
                                 (landmarks.part(44).x, landmarks.part(44).y),
                                 (landmarks.part(45).x, landmarks.part(45).y),
                                 (landmarks.part(46).x, landmarks.part(46).y),
                                 (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [right_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [right_eye_region], 255)
    right_eye = cv2.bitwise_and(gray, gray, mask=mask)

    left_eye_center = (landmarks.part(36).x + landmarks.part(39).x) // 2, (landmarks.part(36).y + landmarks.part(39).y) // 2
    right_eye_center = (landmarks.part(42).x + landmarks.part(45).x) // 2, (landmarks.part(42).y + landmarks.part(45).y) // 2

    if left_eye_center[0] < width // 2:
        return "Focused"
    else:
        return "Not Focused"

# Initialize the webcam
cap = cv2.VideoCapture(0)
# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained shape predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def show_frame(notebook=False):
    ret, frame = cap.read()
    if not ret:
        return False

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 64, 100), -1)

        gaze = detect_gaze(landmarks, frame, gray)
        cv2.putText(frame, gaze, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if notebook:
        # Convert the frame to an image
        _, img = cv2.imencode('.jpg', frame)
        img = PIL.Image.open(BytesIO(img))

        # Display the image in the notebook
        clear_output(wait=True)
        display(img)
    else:
        # Display the frame in a window
        cv2.imshow('Gaze Detection', frame)

    return True

# Run the frame display function
try:
    while True:
        if not show_frame(notebook=False):  # Change to True if running in Jupyter Notebook
            break
        time.sleep(0.1)  # Pause for a short time to control frame rate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Gaze Detection', cv2.WND_PROP_VISIBLE) < 1:
            break
except KeyboardInterrupt:
    print("Interrupted by user")

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
