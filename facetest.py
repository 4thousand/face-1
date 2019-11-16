import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainningData.yml")
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]

        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if(Id == 1):
            Id = "big"
        elif(Id == 2):
            Id = "boss"
        elif(Id == 3):
            Id = "P' noi"
        elif(Id == 4):
            Id = "kanan"
        else:
            Id = "Unknow"
        cv2.waitKey(100)
        cv2.putText(img, str(Id), (x, y+h), font, 1, (130, 50, 200), 2)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
