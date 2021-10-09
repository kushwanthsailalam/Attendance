import cv2
import numpy as np
import face_recognition

imgKush = face_recognition.load_image_file('Images/Kush.jpg')
imgKush = cv2.cvtColor(imgKush,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/Kushh.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgKush)[0]
encodeKush = face_recognition.face_encodings(imgKush)[0]
cv2.rectangle(imgKush,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,255,255),2)

faceLoc = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,255,255),2)

results = face_recognition.compare_faces([encodeKush],encodeTest)
faceDis =face_recognition.face_distance([encodeKush],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
cv2.imshow('Kush',imgKush)
cv2.imshow('Kushh',imgTest)
cv2.waitKey(0)

