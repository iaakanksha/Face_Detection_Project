import numpy as np
import cv2
import face_recognition

#image loading1
imgElon = face_recognition.load_image_file('AttendenceImages/ELON MUSK2.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
#image loading2
imgElonTest = face_recognition.load_image_file('AttendenceImages/Elon test.jpg')
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

#face detection
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)
#print(faceloc)#it return four values which are the coordinates of rectangle
#face detection for test image
faceLocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,0), 2)

#comparing the faces and finding the distance between them
"""If the value return outs to be true for the following code that means it has find bothe the faces similar
In compare faces image in [] is for compare and the other image is for to compare with the other image which is in []
 It usually takes images in list but since I don't have much images i used this"""
results = face_recognition.compare_faces([encodeElon], encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon], encodeElonTest)
print(results, faceDis)
cv2.putText(imgElonTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) #since the faceDis is an array not a value we have to put[index number of value] in round function

cv2.imshow('Elon musk', imgElon)
cv2.imshow('Elon Test', imgElonTest)
cv2.waitKey(0)
