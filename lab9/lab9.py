import cv2
import imutils
import numpy as np
zdjecie = cv2.imread('zdjecie.png')

cv2.imshow("Metallica",zdjecie)
cv2.waitKey(0)

detail = zdjecie[300:500,500:700]

cv2.imshow("Metallica",detail)
cv2.waitKey(0)

resized = cv2.resize(zdjecie,(300,300))

cv2.imshow("Metallica",resized)
cv2.waitKey(0)

h,w=zdjecie.shape[0:2]
matrix = cv2.getRotationMatrix2D((w//2,h//2),-45,1.0)
rotated = cv2.warpAffine(zdjecie,matrix,(w,h))
cv2.imshow("Metallica",rotated)
cv2.waitKey(0)

blurred = cv2.blur(zdjecie,(10,10))
resized = imutils.resize(zdjecie,width=300)
blurred = imutils.resize(blurred,width=300)
sum = np.hstack((resized,blurred))
cv2.imshow("Metallica",sum)
cv2.waitKey(0)

img_copy = zdjecie.copy()
cv2.rectangle(img_copy,(100,200),(300,350),(0,255,255),3)
cv2.line(img_copy,(100,100),(400,50),(0,0,255),5)
points = np.array([[400,400],[510,100],[20,300]])
cv2.polylines(img_copy,np.int32([points]),1,(255,0,0))
cv2.circle(img_copy,(500,450),30,(0,255,0),7)
cv2.imshow("Metallica",img_copy)
cv2.waitKey(0)


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(zdjecie,'Wojciech Jachimowski',(135,450),font,1,(0,0,0),2,cv2.LINE_4)
cv2.imshow("Metallica",zdjecie)
cv2.waitKey(0)