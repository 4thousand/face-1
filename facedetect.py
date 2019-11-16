import cv2
import numpy as np
import json
import yaml

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainningData.yml")
Id = 0
# name = "recognizer\data.yml"
# with open(name, 'r') as stream:
#     names = yaml.load(stream)
name = "recognizer\dataset.json"
with open(name, 'r') as yaml_in:
    yaml_object = yaml.load(yaml_in) # yaml_object will be a list or a dict
    print(yaml_object['data'])
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    if name:
        print(1)
    else:
        print(5)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
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
        if conf > 50:
            output_dict = [x for x in yaml_object['data'] if x['id'] == Id]
            if output_dict: 

                cv2.putText(
                    img, "name:"+str(output_dict[0]["name"]+str(Id)), (x, y+h+20), font, 0.5, (130, 50, 200), 2)
                cv2.putText(img, str(conf),
                        (x, y+h+50), font, 0.5, (130, 50, 200), 2)
        else:
            cv2.putText(
                img, "unknow", (x, y+h+20), font, 0.5, (130, 50, 200), 2)

    # cv2.PutText(img,str(id),(x,y+h),font,255)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
