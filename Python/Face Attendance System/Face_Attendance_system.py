import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

prabin_image = face_recognition.load_image_file("faces/prabin.jpg")
prabin_encoding = face_recognition.face_encodings(prabin_image)[0]

# pooja_image = face_recognition.load_image_file("faces/pooja.jpg")
# pooja_encoding = face_recognition.face_encodings(pooja_image)[0]
#
# somnath_image = face_recognition.load_image_file("faces/somnath.jpg")
# somnath_encoding = face_recognition.face_encodings(somnath_image)[0]
#
# ranjan_image = face_recognition.load_image_file("faces/ranjan.jpg")
# ranjan_encoding = face_recognition.face_encodings(ranjan_image)[0]

known_face_encodings = [prabin_encoding] #, pooja_encoding , somnath_encoding ,ranjan_encoding]
known_faces = ["Prabin" , "Pooja" , "Somanth" , "Ranjan"]

students = known_faces.copy()

face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date},.csv","w+",newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25 , fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame , face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings , face_encoding)

        best_match_indexes = np.argmin(face_distance)
        if(matches[best_match_indexes]):
            name = known_faces[best_match_indexes]

        if name in known_faces:
            font = cv2.FONT_HERSHEY_SIMPLEX  # Correct font face assignment
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)  # Red color in BGR format
            thickness = 3
            lineType = 2

            # Correct usage of cv2.putText()
            cv2.putText(frame, name + " is Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                        lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()