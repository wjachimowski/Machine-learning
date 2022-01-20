import cv2
import numpy
#import inutils
import time

# One face

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
scaling_factor = 0.5
frame = cv2.imread('zd1.jpg')
frame = cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
face_rect = face_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5)

for(x,y,w,h) in face_rect:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

cv2.imshow('zd1.jpg',frame)
cv2.waitKey(0)
print("Found "+str(len(face_rect))+"faces")

# Multiple faces
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
frame = cv2.imread('zd2.jpg')
gray_filter = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#frame = cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
face_rect = face_cascade.detectMultiScale(gray_filter,7,4)

for(x,y,w,h) in face_rect:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_filter[y:y+h,x:x+w]
    roi_color = frame[y:y + h, x:x + w]
    smile = smile_cascade.detectMultiScale(roi_gray)
    eye = eye_cascade.detectMultiScale(roi_gray)
    for(sx,sy,sw,sh) in smile:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),1)
    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 2550), 1)

cv2.imshow('zd2.jpg',frame)
cv2.waitKey(0)
print("Found "+str(len(face_rect))+"faces")



cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rect = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    if cv2.waitKey(1) == ord('0'):
        break
    cv2.imshow('video.mp4', frame)

cap.release()
cv2.destroyAllWindows()


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('video2.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame,(800,500))
    boxes,weights = hog.detectMultiScale(frame,winStride=(8,8))
    boxes = numpy.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])

    for (xa,ya,xb,yb) in boxes:
        cv2.rectangle(frame,(xa,ya),(xb,yb),(0,255,0),1)
    print("Number of people on frame "+ str(len(boxes)))
    cv2.imshow('vide.mp4',frame)

    if cv2.waitKey(1) == ord('0'):
        break
cap.release()
cv2.destroyAllWindows()


