import cv2 as cv
import time
import numpy as np 
cimport numpy as cnp 
from cython.view cimport array as cvarray
import cython
from threading import Thread 
import concurrent.futures
#from libc.stdio cimport printf

cap = cv.VideoCapture(0) # <class 'cv2.VideoCapture'>)
logo = cv.imread('logo.png', cv.IMREAD_GRAYSCALE) # <class 'numpy.ndarray'>)
logo = cv.Canny(logo, 50, 200) # <class 'numpy.ndarray'>)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def pyramid(cnp.ndarray img, float scale=0.5, (int, int) min_size=(32,32)):
    
    yield img 
    
    
    while True:
        img = cv.resize(img, None,fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
        if (img.shape[0] < min_size[0]) and (img.shape[1] < min_size[1]):
            break
        yield img 


cdef list imagenes = []

for (i, resized) in enumerate(pyramid(logo, scale=0.95)):
    imagenes.append(resized) # dtype=uint8)]


cdef run():
    cdef int h, w
    cdef float min_val, max_val, valor_max
    cdef (int, int) localidad_max, min_loc, max_loc, top_left, bottom_right ,tipo

    #cdef int[:, :] res
    #cdef cnp.ndarray[cnp.float32_t, ndim=2] res
    #cdef double [:, :] res


    tiempo1 = time.time()
    while(True):
        localidad_max = (0,0) #  <class 'tuple'>)

        ret, frame = cap.read() # Frame: <class 'numpy.ndarray'>) ; ret : <class 'bool'>)

        #PASAR FRAME A BLANCO Y NEGRO
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #APLICAR EDGE DETECTION PARA QUE SEA MAS PRECISO 
        img = cv.Canny(img, 50,200)

        valor_max = 0.0
        

        tipo = (0,0)
        for i in range(len(imagenes)):
            imagen = imagenes[i] #<class 'numpy.ndarray'>)  dtype=float32
            #print('imagen', imagen)

            res = cv.matchTemplate(img,imagen, cv.TM_CCOEFF_NORMED) # <class 'numpy.ndarray'>) dtype=float32))
            #print('res', res)

            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            #min_val, max_val = cv.minMaxLoc(res)[0,1]
            #min_loc, max_loc = cv.minMaxLoc(res)[2,3]
            # min_val, max_val  <class 'float'>)
            # min_loc, max_loc <class 'tuple'>)

            if max_val > valor_max:
                valor_max = max_val
                localidad_max = max_loc
                tipo = imagen.shape

        h,w = tipo
        top_left = localidad_max
        bottom_right = (top_left[0] + w, top_left[1] + h) # <class 'tuple'>

        frames = round(0.6 / (time.time() - tiempo1), 2)
        tiempo1 = time.time()
        cv.putText(frame,"FPS: " + str(frames), (0,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        if valor_max > 0.4:
            cv.rectangle(frame,top_left, bottom_right, (0,255,0), 3)

        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

run()

cap.release()
cv.destroyAllWindows()