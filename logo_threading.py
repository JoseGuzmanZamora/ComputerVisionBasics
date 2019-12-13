import cv2 as cv
import time
from threading import Thread 
import concurrent.futures
import video
import numpy as np 

cap = cv.VideoCapture(0)

logo = cv.imread('./logopeqe.png', cv.IMREAD_GRAYSCALE)
logo = cv.Canny(logo, 50, 200)

imagenes = []
for (i, resized) in enumerate(video.pyramid(logo, 0.95)):
    imagenes.append(resized)
print(len(imagenes))

#funcion que va a ser llamada por threading 
def piramide_match(result, indice, frame_actual, inicio, final):
    valor_max = 0
    localidad_max = (0,0)
    tipo = (0,0)
    for i in range(inicio, final + 1):
        imagen_template = imagenes[i]
        res = cv.matchTemplate(frame_actual,imagen_template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if max_val > valor_max:
            valor_max = max_val
            localidad_max = max_loc
            tipo = imagen_template.shape
    result[indice] = [valor_max, localidad_max, tipo]

tiempo1 = time.time()
cantidad = 7
while(True):
    ret, frame = cap.read()

    #PASAR FRAME A BLANCO Y NEGRO
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #APLICAR EDGE DETECTION PARA QUE SEA MAS PRECISO 
    img = cv.Canny(img, 50,200)

    resultados = [None] * cantidad
    threads = [None] * cantidad
    division = round(len(imagenes) / cantidad)
    for k in range(0,cantidad):
        threads[k] = Thread(target=piramide_match, args=(resultados, k, img, k * division , ((k * division) + division) - 1))
        threads[k].start()
    
    for i,thread in enumerate(threads):
        thread.join()

    
    order = sorted(resultados, key=lambda x: x[0])
    best = order[cantidad - 1]
    h,w = best[2]
    top_left = best[1]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if best[0] > 0.4:
        cv.rectangle(frame,top_left, bottom_right, (0,255,0), 3)
    

    frames = round(0.6 / (time.time() - tiempo1))
    tiempo1 = time.time()
    cv.putText(frame,str(frames), (0,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv.imshow("Camera", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()
    


