import cv2 as cv 
import time
import numpy as np

cap = cv.VideoCapture(0)

while(True):

    ret, frame = cap.read()

    #TRANSFORMACIONES 
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(frame, 100, 200)

    cv.imshow("Result Image", edges)

    window_name = 'Original'

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    cv.imshow(window_name,frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()