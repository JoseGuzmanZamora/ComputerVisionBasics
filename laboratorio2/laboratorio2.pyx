#José Alejandro Guzmán Zamora 
#Carlos Cuj Cuj 

import cv2 as cv
import time
from threading import Thread 
import concurrent.futures
import numpy as np   # Normal NumPy import
cimport numpy as cnp # Import for NumPY C-API
import cython

cap = cv.VideoCapture(0) # <class 'cv2.VideoCapture'>)
logo = cv.imread('logo.png', cv.IMREAD_GRAYSCALE) # <class 'numpy.ndarray'>)
logo = cv.Canny(logo, 50, 200) # <class 'numpy.ndarray'>)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def pyramid(cnp.ndarray img, cnp.float scale=0.5, (int, int) min_size=(32,32)):
    """
        Build a pyramid for an image until min_size
        dimensions are reached. 
        
        Args:
            img (numpy array): source image 
            scale(float): scaling factor
            min_size(tuple): minimum size of pyramid top level 
        Returs:
            Pyramid generator
    """
    yield img 
    
    while True:
        img = cv.resize(img, None,fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
        if (img.shape[0] < min_size[0]) and (img.shape[1] < min_size[1]):
            break
        yield img 

cdef list imagenes = [] #<class 'list'>

for (i, resized) in enumerate(pyramid(logo, scale=0.95)):
    imagenes.append(resized)


#funcion que va a ser llamada por threading 
def piramide_match(list result, cnp.int indice, cnp.ndarray frame_actual, cnp.int inicio, cnp.int final):
    
    '''
    piramide_match:

    Aplica la funcion de template matching en un conjunto de imagenes obtenidos de la funcion
    pyramid, estas imagenes almacenadas en un array.

    Args:
    result (list): Lista de imagenes obtenidas por medio de pyramid

    indice (cnp.int): ubicacion donde se ingresa al resultado 

    frame_actual (cnp.array): frame en procesamiento al momento, obtenido por la camara

    inicio (cnp.inicio): Indice inicial del array donde se encuentra la imagen utilizara para el template matching

    final (cnp.final): Indice final del array donde se encuentra la imagen utilizara para el template matching
    
    return (void):
        Modificaciones a la variable de resultados por referencia. 

    '''
    
    cdef float min_val, max_val, valor_max
    cdef (int, int) localidad_max, min_loc, max_loc

    valor_max = 0 #<class 'int'>
    localidad_max = (0,0) # <class 'tuple'> int
    tipo = (0,0) #<class 'tuple'> int 

    for i in range(inicio, final + 1):
        imagen_template = imagenes[i] # <class 'numpy.ndarray'>
        
        res = cv.matchTemplate(frame_actual,imagen_template, cv.TM_CCOEFF_NORMED)
        # res <class 'numpy.ndarray'>
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        #('min_val', max_val <class 'float'>
        #('min_loc', max_loc <class 'tuple'>


        if max_val > valor_max:
            valor_max = max_val
            localidad_max = max_loc
            tipo = imagen_template.shape
    result[indice] = [valor_max, localidad_max, tipo] #<class 'list'>


#cdef float tiempo1, frames
cdef int cantidad, division, h, w
cdef list resultados, threads, best
cdef (int, int) top_left, bottom_right

tiempo1 = time.time()
cantidad = 7 #<class 'int'>,

while(True):

    ret, frame = cap.read()

    #PASAR FRAME A BLANCO Y NEGRO
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #APLICAR EDGE DETECTION PARA QUE SEA MAS PRECISO 
    img = cv.Canny(img, 50,200)

    resultados = [None] * cantidad #<class 'list'>
    threads = [None] * cantidad # <class 'list'>
    division = round(len(imagenes) / cantidad) # <class 'int'>


    for k in range(0,cantidad):
        threads[k] = Thread(target=piramide_match, args=(resultados, k, img, k * division , ((k * division) + division) - 1))
        threads[k].start()
    
    for i,thread in enumerate(threads):
        thread.join() #<class 'threading.Thread'>


    order = sorted(resultados, key=lambda x: x[0]) #<class 'list'>
    best = order[cantidad - 1] #<class 'list'>
    h,w = best[2] #<class 'int'>

    top_left = best[1] #<class 'tuple'> int
    bottom_right = (top_left[0] + w, top_left[1] + h) #<class 'tuple'> int

    if best[0] > 0.4:
        cv.rectangle(frame,top_left, bottom_right, (0,255,0), 3)
    
    frames = round(0.6 / (time.time() - tiempo1), 2)
    tiempo1 = time.time() 

    cv.putText(frame,"FPS: " + str(frames), (0,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv.imshow("Camera", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()