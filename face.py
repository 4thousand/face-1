import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainningData.yml")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, rgb = img.read()
    grays = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grays, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = grays[y:y+h, x:x+w]
        roi_color = grays[y:y+h, x:x+w]
        Id, conf = recognizer.predict(grays[y:y+h, x:x+w])
        if(Id == 1):
            Id = "big"
        elif(Id == 2):
            Id = "boss"
        elif(Id == 3):
            Id = "pe noi"
        elif(Id == 4):
            Id = "kanan"
        else:
            Id = "Unknow"
        print(str(Id))
        cv2.putText(rgb, str(Id), (x, y+h), font, 1, (130, 50, 200), 2)
    cv2.imshow("Face", rgb)
cv2.destroyAllWindows()


cv2.waitKey(0)
cv2.destroyAllWindows()
