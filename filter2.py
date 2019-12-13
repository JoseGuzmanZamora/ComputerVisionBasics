import cv2 as cv 
import numpy as np
import time

cap = cv.VideoCapture(0)
divisiones = 800
offset = 5
comienzo = True

while(True):
    ret, frame = cap.read()

    '''img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    alto = img.shape[0]
    largo = img.shape[1]
    copia = img.copy()
    
    division = int(largo / divisiones)
    contador = 0

    for i in range(len(img)):
        for j in range(len(img[i])):
            if comienzo:
                if i > offset:
                    img[i, j] = copia[i - offset, j]
                if contador >= division:
                    comienzo = False
                    contador = 0
                contador += 1
            else:
                if i < (alto - offset - 1):
                    img[i , j] = copia[i + offset, j]
                if contador >= division:
                    comienzo = True
                    contador = 0
                contador += 1'''
    img = frame
    divisiones = 20
    offset = 10
    comienzo = True
    alto = img.shape[0]
    largo = img.shape[1]
    copia = img.copy()
    
    division = int(alto / divisiones)
    contador = 0

    for i in range(len(img)):
        for j in range(len(img[i])):
            if comienzo:
                if j > offset:
                    img[i, j] = copia[i, j - offset]
                    img[i,j] = img[i,j] * 2
                if contador >= division:
                    comienzo = False
                    contador = 0
            else:
                if j < (largo - offset - 1):
                    img[i , j] = copia[i, j + offset]
                    img[i, j] = img[i, j] / 2
                if contador >= division:
                    comienzo = True
                    contador = 0
        contador += 1

    cv.imshow("Shadow",img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()