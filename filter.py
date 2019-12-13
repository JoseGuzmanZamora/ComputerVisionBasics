import cv2 as cv 
import numpy as np
import time

cap = cv.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    frame2 = cv.Canny(frame, 150, 200)
    trasladar = frame.shape[1]
    difj = int(trasladar / 4)

    for i in range(len(frame2)):
        for j in range(len(frame2[i])):
            if(frame2[i, j] == 255):
                if(j + difj) < trasladar:
                    frame[i : i + 3, j + difj] = [255,0,0]
                if(j - difj) > 0:
                    frame[i : i + 3, j - difj] = [255,0,0]

                

                valor = frame[i + 4, j]
                valor2 = frame[i, j - 1]
                color = [255, 0,0]

                if not (valor[0] == 255 and valor[1] == 0 and valor[2] == 0):
                    if(j + difj) < trasladar:
                        frame[i + 4 : i + 7, j + difj] = [0,255,0]
                    if(j - difj) > 0:
                        frame[i + 4 : i + 7, j - difj] = [0,255,0]

    cv.imshow("Tri",frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()