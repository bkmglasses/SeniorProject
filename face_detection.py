import cv2
import numpy as np
import urllib.request
from keras.models import load_model  # TensorFlow is required for Keras to work
import pandas as pd
import face_recognition
import pytesseract  # Import for Tesseract OCR
from datetime import datetime
import os
import gc  # Import for garbage collection
import tensorflow as tf
import threading

# Ensure that TensorFlow uses the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to use CPU instead of GPU

# Path to Tesseract (this may vary depending on your system and installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\marya\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Adjust path for Windows


# For Linux, it might be '/usr/bin/tesseract'

# Mark attendance function for face recognition
def markAttendance(name):
    try:
        df = pd.read_csv(attendance_file)
        if name not in df["Name"].values:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            new_entry = pd.DataFrame({"Name": [name], "Time": [dtString]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(attendance_file, index=False)
    except Exception as e:
        print(f"Error saving attendance: {e}")


# Set up paths and URLs
path = r'D:\attendace\attendace\image_folder'  # Path to known faces images folder
url = 'http://192.168.88.15/cam-hi.jpg'  # ESP32-CAM streaming URL
attendance_folder = os.path.join(os.getcwd(), 'attendace')
attendance_file = os.path.join(attendance_folder, "Attendance.csv")

if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)

# Load images and encode faces for face recognition
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])


# Encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Load object detection model
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


# Function to process each frame of the stream
def process_frame(img):
    try:
        # Resize the image for object detection
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Object detection
        image_input = np.asarray(img_rgb, dtype=np.float32).reshape(1, 224, 224, 3)
        image_input = (image_input / 127.5) - 1
        prediction = model.predict(image_input)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Face detection using face_recognition
        img_rgb_resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(img_rgb_resized, model="hog")
        encodesCurFrame = face_recognition.face_encodings(img_rgb_resized, faces)

        processed_faces = set()  # To avoid duplicate attendance marking
        for encodeFace, faceLoc in zip(encodesCurFrame, faces):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis) if faceDis.size > 0 else -1
            name = classNames[matchIndex].upper() if matchIndex != -1 and faceDis[matchIndex] < 0.6 else "UNKNOWN"

            if name != "UNKNOWN" and name not in processed_faces:
                markAttendance(name)  # Ensure this function is defined before calling
                processed_faces.add(name)

            # Draw face rectangle and label
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
            color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Print the name of the detected face
            print(f"Recognized face: {name}")

        # Display object detection result
        if confidence_score >= 0.91:
            print(f"Object Detected: {class_name.strip()}  Confidence: {np.round(confidence_score * 100, 2)}%")

        # **Text Detection (OCR using Tesseract)**
        # Convert the image to grayscale for better OCR accuracy
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_img)

        # Check if text is detected
        if text.strip() != "":
            print("TEXT DETECTED")

        # Display both face detection, object detection, and OCR results
        cv2.imshow("ESP32-CAM Stream", img)
        gc.collect()  # Garbage collection to free memory
    except Exception as e:
        print(f"Error processing frame: {e}")


# Webcam or ESP32-CAM Stream
while True:
    try:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)

        process_frame(img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error processing webcam stream: {e}")
        break  # Exit the loop on error

cv2.destroyAllWindows()
