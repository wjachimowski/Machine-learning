import cv2
import numpy as np
import math

cap = cv2.VideoCapture('zad.mp4')
l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(l)

cap = cv2.VideoCapture('zad.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(15,15),0)
    if cv2.waitKey(1) == ord('0'):
        break
    cv2.putText(frame, 'Wojciech Jachimowski', (1000, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow('zad.mp4', frame)

cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture('zad.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    canny_low_thresh_hold =20
    canny_high_thresh_hold =100
    def canny(frame, low_thresh_hold, high_thresh_hold):
        return cv2.Canny(frame, low_thresh_hold,high_thresh_hold)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(15,15),0)

    blur_canny = canny(frame,canny_low_thresh_hold,canny_high_thresh_hold)
    cv2.putText(blur_canny, 'Wojciech Jachimowski', (1000, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow('zad.mp4', blur_canny)
    if cv2.waitKey(1) == ord('0'):
        break


cap.release()
cv2.destroyAllWindows()



cap = cv2.VideoCapture('zad.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    canny_low_thresh_hold =20
    canny_high_thresh_hold =100
    def canny(frame, low_thresh_hold, high_thresh_hold):
        return cv2.Canny(frame, low_thresh_hold,high_thresh_hold)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(15,15),0)
    blur_canny = canny(frame,canny_low_thresh_hold,canny_high_thresh_hold)
    h=1080
    w=1920
    x=600
    y=0
    blur_canny_temp = blur_canny[x:x + h,y:y+w]
    blur_canny = np.zeros_like(blur_canny)
    blur_canny[x:x +h,y:y+w] = blur_canny_temp
    cv2.putText(blur_canny, 'Wojciech Jachimowski', (1000, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow('zad.mp4', blur_canny)
    if cv2.waitKey(1) == ord('0'):
        break


cap.release()
cv2.destroyAllWindows()



cap = cv2.VideoCapture('zad.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    canny_low_thresh_hold =20
    canny_high_thresh_hold =100
    def canny(frame, low_thresh_hold, high_thresh_hold):
        return cv2.Canny(frame, low_thresh_hold,high_thresh_hold)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(15,15),0)
    blur_canny = canny(frame,canny_low_thresh_hold,canny_high_thresh_hold)
    h=1080
    w=1920
    x=600
    y=0
    blur_canny_temp = blur_canny[x:x + h,y:y+w]
    blur_canny = np.zeros_like(blur_canny)
    blur_canny[x:x +h,y:y+w] = blur_canny_temp

    dst = cv2.Canny(blur_canny,50,200,None,3)
    cdst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines =cv2.HoughLines(dst,1,np.pi/180,150,None,0,0)
    if lines is not None:
        for i in range(0,len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a=math.cos(theta)
            b=math.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (0,0,255),3,cv2.LINE_AA)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0,len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP,(l[0],l[1]),(l[2],l[3]),(0,0,255),3,cv2.LINE_AA)

    cv2.putText(cdstP, 'Wojciech Jachimowski', (1000, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    #video = cv2.addWeighted(frame,0.8,cdstP,1,0)
    cv2.imshow('zad.mp4', cdstP)
    #cv2.imshow('zad.mp4', frame)
    if cv2.waitKey(1) == ord('0'):
        break



    ######

cap.release()
cv2.destroyAllWindows()