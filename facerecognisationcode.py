import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

AbdulKalam_image = face_recognition.load_image_file(r"C:\Users\Tanushree\Desktop\Face_recognization\photos\AbdulKalam.jpeg")
AbdulKalam_encoding = face_recognition.face_encodings(AbdulKalam_image)[0]

NarendraModi_image = face_recognition.load_image_file(r"C:\Users\Tanushree\Desktop\Face_recognization\photos\NarendraModi.jpeg")
NarendraModi_encoding = face_recognition.face_encodings(NarendraModi_image)[0]

ViratKoli_image = face_recognition.load_image_file(r"C:\Users\Tanushree\Desktop\Face_recognization\photos\ViratKoli.png")
ViratKoli_encoding = face_recognition.face_encodings(ViratKoli_image)[0]

KiaraAdvani_image = face_recognition.load_image_file(r"C:\Users\Tanushree\Desktop\Face_recognization\photos\KiaraAdvani.jpeg")
KiaraAdvani_encoding = face_recognition.face_encodings(KiaraAdvani_image)[0]

DishaPatani_image = face_recognition.load_image_file(r"C:\Users\Tanushree\Desktop\Face_recognization\photos\DishaPatani.jpg")
DishaPatani_encoding = face_recognition.face_encodings(DishaPatani_image)[0]

known_face_encoding = [
    AbdulKalam_encoding,
    NarendraModi_encoding,
    ViratKoli_encoding,
    KiaraAdvani_encoding,
    DishaPatani_encoding
]

known_faces_names = [
    "AbdulKalam",
    "NarendraModi",
    "ViratKoli",
    "KiaraAdvani",
    "DishaPatani"
]

students = known_faces_names.copy()
attendance_recorded = set()  # Set to store students whose attendance has been recorded

video_capture = cv2.VideoCapture(0)

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create CSV file and write header
csv_filename = current_date + '.csv'
with open(csv_filename, 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.15, fy=0.15)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index] 

            if name in students and name not in attendance_recorded:
                attendance_recorded.add(name)  # Add to set to mark attendance
                current_time = now.strftime("%H-%M-%S") 
                with open(csv_filename, 'a', newline='') as f:
                    lnwriter = csv.writer(f)
                    lnwriter.writerow([name, current_time])
                print(f"{name} Marked Present")

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 6
        right *= 6
        bottom *= 6
        left *= 6

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
