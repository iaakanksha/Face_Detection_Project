import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'AttendenceImages'  # path is created
images = []  # list of images
classNames = []
mylist = os.listdir(path)  # grabbing the list
print(mylist)
# to print all the images from the folder
for cls in mylist:
    currentimage = cv2.imread(f'{path}/{cls}')
    images.append(currentimage)
    classNames.append(os.path.splitext(cls)[0])  # used to print only the names of images without extensions
print(classNames)


# function for encoding finding
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#creating function to mark attendence
def markAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtstring}')

#encoding the known images and making a list of them
encodelistknown = findEncodings(images)
#print(len(encodelistknown))
print("Encoding Complete")

#Initializing the webcam
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)  # reducing the siZe of the image so that it will speed up the process
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #finding the encoding of images from webcam
    faceCurframe = face_recognition.face_locations(imgS)
    encodeCurframe = face_recognition.face_encodings(imgS, faceCurframe)

    for encodeFace, faceloc in zip(encodeCurframe,faceCurframe): #used zip because finding 2 things ina same loop
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
           # print(name)
            y1, x2, y2, x1 = faceloc
            """if the box is not placing around the face correctly its because
            we have scaled down our image by 1/4th
            so to place it correctly we have to myltiply it by 4 for that include this line of code"""
            # y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            markAttendence(name)


        cv2.imshow("Webcam", img)
        cv2.waitKey(1)



