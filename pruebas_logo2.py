import cv2 as cv
import time
import video
import numpy as np 


cap = cv.VideoCapture(0)

logo = cv.imread('./logopeqe.png', cv.IMREAD_GRAYSCALE)
logo = cv.Canny(logo, 50, 200)

imagenes = []
for (i, resized) in enumerate(video.pyramid(logo, 0.95)):
    imagenes.append(resized)


tiempo1 = time.time()
while(True):
    ret, frame = cap.read()

    #PASAR FRAME A BLANCO Y NEGRO
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #APLICAR EDGE DETECTION PARA QUE SEA MAS PRECISO 
    img = cv.Canny(img, 50,200)

    valor_max = 0
    localidad_max = (0,0)
    tipo = (0,0)
    buenos_tipo = []
    buenos_loc = []
    for i in range(len(imagenes)):
        imagen = imagenes[i]
        res = cv.matchTemplate(img,imagen, cv.TM_CCOEFF_NORMED)
        threshold = 0.4
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            buenos_tipo.append(imagen.shape)
            buenos_loc.append(pt)

    for i in range(len(buenos_loc)):
        h,w = buenos_tipo[i]
        top_left = buenos_loc[i]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(frame,top_left, bottom_right, (0,255,0), 3)

    frames = round(0.6 / (time.time() - tiempo1), 2)
    tiempo1 = time.time()
    cv.putText(frame,"FPS: " + str(frames), (0,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()