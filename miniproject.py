import cv2
import numpy as np
import json
import yaml

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainningData.yml")
Id = 0
name = "recognizer\data.yml"
with open(name, 'r') as stream:
    names = yaml.load(stream)
name = "recognizer\data.yml"
with open(name, 'r') as yaml_in:
    yaml_object = yaml.load(yaml_in) # yaml_object will be a list or a dict
    print(yaml_object['data'])

print(output_dict[0]['name'])

font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    if names:
        print(1)
    else:
        print(5)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
#        if(Id==1
#            Id="suriya"
#        elif(Id==2):
#            Id="tong"
#        elif(Id==3):
#            Id="Joe"
#        elif(Id==4):
#            Id="kanan"
#        else:
#            Id="Unknow"
        output_dict = [x for x in yaml_object['data'] if x['id'] == Id]
        cv2.putText(
            img, "name:"+str(names[Id]["name"]), (x, y+h+20), font, 0.5, (130, 50, 200), 2)
        cv2.putText(img, "age:"+str(names[Id]["age"]),
                    (x, y+h+40), font, 0.5, (130, 50, 200), 2)
        cv2.putText(img, "age:"+str(conf),
                    (x, y+h+50), font, 0.5, (130, 50, 200), 2)
    # cv2.PutText(img,str(id),(x,y+h),font,255)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
