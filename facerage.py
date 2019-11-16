import cv2
import numpy as np
import yaml
import json
name = "recognizer\dataset.json"
with open(name, 'r') as yaml_in:
    yaml_object = yaml.load(yaml_in) # yaml_object will be a list or a dict
    print(yaml_object['data'])
List_length = len(yaml_object['data'])

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

names = input('enter name')
yaml_object['data'].append({
      "id": List_length+1,
      "name": names,
      "phone": "00-000",ice
      "age": 23
    })
with open(name,'w') as yaml_out:
    json.dump(yaml_object,yaml_out,indent=4)
sampleNum = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cv2.imshow("faces",gray)
    for(x, y, w, h) in faces:
        sampleNum = sampleNum+1
        cv2.imwrite("dataSet/User."+str(List_length+1)+"." +
                    str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 30:
        break
        
cap.release()
cv2.destroyAllWindows()
