import cv2
import dlib
import time

# initialize the Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# initialize the Haar Cascade classifier for eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initialize the dlib facial landmark predictor for eye aspect ratio (EAR) calculation
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# initialize the video stream and wait for the camera to warm up
video_capture = cv2.VideoCapture(0)
time.sleep(1.0)

# define a function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # calculate the distance between the vertical eye landmarks
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    # calculate the distance between the horizontal eye landmarks
    C = euclidean_distance(eye[0], eye[3])
    # calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# define a function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# define the threshold for determining if the eyes are closed
ear_threshold = 0.25

# initialize the frame counters and time variables
frame_counter = 0
blink_counter = 0
distracted_counter = 0
start_time = time.time()

# loop over the frames from the video stream
while True:
    # grab the frame from the video stream, resize it, and convert it to grayscale
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # loop over the faces
    for (x, y, w, h) in faces:
        # crop the face region of interest and convert it to grayscale
        roi_gray = gray[y:y+h, x:x+w]

        # detect eyes in the grayscale face region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        # loop over the eyes
        for (ex, ey, ew, eh) in eyes:
            # calculate the eye landmarks using dlib
            shape = predictor(frame, dlib.rectangle(x+ex, y+ey, x+ex+ew, y+ey+eh))
            left_eye = [(shape.part(36).x, shape.part(36).y), (shape.part(37).x, shape.part(37).y),
                        (shape.part(38).x, shape.part(38).y), (shape.part(39).x, shape.part(39).y),
                        (shape.part(40).x, shape.part(40).y), (shape.part(41).x, shape.part(41).y)]
