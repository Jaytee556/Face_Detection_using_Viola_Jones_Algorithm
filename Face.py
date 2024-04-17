import numpy as np
import cv2

# Step 1: Identify the webcam
webcam = cv2.VideoCapture(0) # Local webcam - 0, External webcam - 1

# Load the face cascade classifier
face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_name)

# Eye detection
eye_cascade_name = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_name)

# Smile detection
smile_cascade_name = cv2.data.haarcascades + 'haarcascade_smile.xml'
smile_cascade = cv2.CascadeClassifier(smile_cascade_name)

# Create a function to capture frames from the webcam and detect faces
def detect(gray, frame):
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Eye Detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # Actual eye detection
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 100), 3)
        
        # Smile Detection
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (100, 55, 160), 3)
        
    return frame

# Step 2: Switch on the webcam
while True:
    ret, frame = webcam.read() # switch on the webcam 
    
    if not ret:  
        print("Error: Frame is empty")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_frame = detect(gray, frame)
    
    cv2.imshow('Face Detection', detected_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
