#José Alejandro Guzmán Zamora 

from datetime import datetime
import cv2 as cv
import numpy as np
from threading import Thread
from queue import Queue
import time
import multiprocessing as mp 


# APLICACION DE EFECTO 
def effect(img):
    """ 
    Aplicacion de transformaciones básicas sobre imagen.

    Divide la imagen de manera horizontal y dezplaza cada división verticalmente. 

    Args:
        - img (numpy array): Imagen a procesar

    Returns:
        - result_img (numpy array): Imagen procesada con las transformaciones 
    """
    '''divisiones = 16
    offset = 5
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
                    img[i,j] = img[i,j] * 1.5
                if contador >= division:
                    comienzo = False
                    contador = 0
            else:
                if j < (largo - offset - 1):
                    img[i , j] = copia[i, j + offset]
                    img[i, j] = img[i, j] * 1.5
                if contador >= division:
                    comienzo = True
                    contador = 0
        contador += 1'''
    hola = np.full_like(img, 5)
    img = np.multiply(img, hola)
    return img

def effect2(mp_array, r, c):
    """ 
    Aplicacion de transformaciones básicas sobre imagen.

    Divide la imagen de manera horizontal y dezplaza cada división verticalmente. 

    Args:
        - img (numpy array): Imagen a procesar

    Returns:
        - result_img (numpy array): Imagen procesada con las transformaciones 
    """

    img = np.reshape(np.frombuffer(mp_array, dtype=np.uint8),(r,c,-1))
    img[:,:] = img[:,:] * 5
    '''return img
    divisiones = 2
    offset = 5
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
                    img[i,j] = img[i,j] * 1.5
                if contador >= division:
                    comienzo = False
                    contador = 0
            else:
                if j < (largo - offset - 1):
                    img[i , j] = copia[i, j + offset]
                    img[i, j] = img[i, j] * 1.5
                if contador >= division:
                    comienzo = True
                    contador = 0
        contador += 1'''

# "SERIAL" NO SE APLICA THREAD NI MULTIPROCESSING 
def serial(src):
    """ 
    Captura y muestra de frames de manera serial. 

    Utilizando cv.VideoCapture() se muestra el video o fuente de imagenes con el efecto determinado. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """

    contador = 0
    cap = cv.VideoCapture(src)
    new_frame, frame = cap.read()
    
    while new_frame:
        frame = effect(frame)
        cv.imshow('SERIAL', frame)
        cv.moveWindow("SERIAL", 0, 0)
        contador += 1
        new_frame, frame = cap.read()

        if cv.waitKey(1) == ord("q"):
            new_frame = False

    return contador

# THREADED WINDOW, THREADING A LA VENTANA DE SALIDA 
class ThreadedWindow:
    """
    Muestra de Frames. 
    Se aplica Threading a la funcion de imshow. 
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
        self.frames = 0

    def __del__(self):
        print('Threaded Window : {0}'.format(self.frames))

    def start(self):
        Thread(target=self.show,args=()).start()

    def update(self, new_frame):
        self.frame = new_frame

    def show(self):
        while(not self.stopped):
            cv.imshow("Threaded Window", self.frame)
            cv.moveWindow("Threaded Window", 0, 0)
            self.frames += 1
            if cv.waitKey(1) == ord("q"):
                self.stop()
                break    

    def stop(self):
        self.stopped = True

def threaded_window(src):
    """ 
    Captura y muestra de frames con la clase ThreadedWindow. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """
    cap = cv.VideoCapture(src)
    new_frame, frame = cap.read()
    frame = effect(frame)

    threaded_window_obj = ThreadedWindow(frame)
    threaded_window_obj.start()

    contador = 1
    while new_frame and not threaded_window_obj.stopped:
        new_frame, frame = cap.read()

        if not new_frame:
            threaded_window_obj.stop()
            break 
        else:
            frame = effect(frame)
            threaded_window_obj.update(frame)
            contador += 1
    return contador

# THREADED READ, THREADING A LA LECTURA DEL FRAME 
class ThreadedRead:
    """
    Captura de Frames. 
    Se aplica Threading a la función que utiliza cap.read(). 
    """
    
    def __init__(self, src=0):
        self.cap = cv.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.frames = 1

    def __del__(self):
        print(' Threaded Read : {0}'.format(self.frames))

    def start(self):
        Thread(target=self.capture,args=()).start()

    def capture(self):
        while True:
            self.ret, self.frame = self.cap.read()
            if self.stopped or not self.ret:
                break
            else:
                self.frames += 1

    def stop(self):
        self.stopped = True

def threaded_read(src):
    """
    Captura y muestra de frames con la clase ThreadedRead. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """

    threaded_read_obj = ThreadedRead(src)
    threaded_read_obj.start()
    contador = 0

    while True:
        frame = threaded_read_obj.frame
        if cv.waitKey(1) == ord("q"):
            threaded_read_obj.stop()
            break
        else:
            if(threaded_read_obj.ret):
                frame = effect(frame)
                cv.imshow("Threaded Read", frame)
                cv.moveWindow("Threaded Read", 0, 0)
                contador += 1 
            else:
                threaded_read_obj.stop()
                break
    return contador

# QUEUE READ, THREADING A LA COLA DE LECTURA 
class QueueRead:
    """
    Captura de Frames. 
    Se aplica Threading a la función que utiliza cap.read() en conjunto con guardado en una cola. 
    """

    def __init__(self, src, queue_size=512):
        self.cap = cv.VideoCapture(src)
        self.enabled = True
        self.Q = Queue(maxsize=queue_size)
        self.frames = 0

    def __del__(self):
        print("Queue Read : {0} frames".format(self.frames))

    def start(self):
        Thread(target=self.capture, args=(), daemon=False).start()
    
    def capture(self):
        while self.enabled:
            if not self.Q.full():
                new_frame, frame = self.cap.read()
                if not new_frame:
                    self.stop()
                    break
                else:
                    self.frames += 1 
                    self.Q.put(frame)
    
    def read(self):
        return self.Q.get()
    def more(self):
        return not self.Q.empty()
    def stop(self):
        self.enabled = False

def queue_read(src):
    """
    Captura y muestra de frames con la clase QueueRead. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """
    queue_read_obj = QueueRead(src)
    queue_read_obj.start()
    time.sleep(0.1)
    frame = queue_read_obj.read()
    contador = 1

    while queue_read_obj.more():
        frame = effect(frame)
        cv.imshow("Queue Read", frame)
        cv.moveWindow("Queue Read", 0, 0)
        if cv.waitKey(1) == ord("q"):
            queue_read_obj.stop()
            break 

        frame = queue_read_obj.read()
        contador += 1 
    queue_read_obj.stop()    
    return contador

# THREADED IO, THREADING TANTO A LA LECTURA COMO A SHOW 
def threaded(src):
    """
    Captura y muestra de frames con la clase QueueRead y ThreadedWindow. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """
    read_obj = QueueRead(src)
    read_obj.start()
    time.sleep(0.1)
    frame = read_obj.read()
    frame = effect(frame)
    show_obj = ThreadedWindow(frame)
    show_obj.start()
    contador = 1 

    while True:
        if(read_obj.more()):
            frame = effect(read_obj.read())
            show_obj.update(frame)
            contador += 1
        else:
            read_obj.stop()
            show_obj.stop()
            break
        if (not read_obj.enabled) or show_obj.stopped:
            read_obj.stop()
            show_obj.stop()
            break
    return contador

def threaded2(src):
    """
    Captura y muestra de frames con la clase ThreadedRead y ThreadedWindow. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """
    read_obj = ThreadedRead(src)
    read_obj.start() 
    show_obj = ThreadedWindow(read_obj.frame)
    show_obj.start()
    contador = 1
    while read_obj.ret:
        contador += 1
        show_obj.update(effect(read_obj.frame))
        if read_obj.stopped or show_obj.stopped:
            break
    read_obj.stop()
    show_obj.stop()
    return contador
  
# MULTIPROCESSING 
def split(img, blocks):
    """ 
    Splits an image into blocks.

    Args:
        img (numpy array): Image to split.
        blocks (int): Number blocks to generate.
    Returns:
        Generator of blocks.
    """
    r,c = img.shape[0:2]
    sz = r//blocks

    for i in range(0,sz*(blocks-1), sz):
        yield img[i:i+sz,:]
    yield(img[i+sz:r,:])

def en_paralelo(frame, n_procs):
    """ 
    Process an image usign n_procs CPUs.

    Args:
        filename (string): Path to input.
        n_procs (int): Number of procesees to spawn.
    Returns:
        Outputs the processed image in a window.
    """
    
    # define containers
    processes = []
    arrays =  []
    for block in split(frame, n_procs):
        r, c = block.shape[0:2]
        pixels = block.ravel()
        shared_array = mp.RawArray('B', pixels)
        p = mp.Process(target=effect2, args=(shared_array, r, c,))
        arrays.append([shared_array,r,c])
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    r_acc = 0
    result = np.zeros_like(frame)
    for array in arrays:
        im_from_proc = np.reshape(np.frombuffer(array[0], dtype=np.uint8), (array[1], array[2], -1))
        result[r_acc:r_acc+array[1],0:array[2],:] = im_from_proc
        r_acc += array[1]
    
    return result

def parallel(src):
    """ 
    La aplicación del efecto se hace en paralelo. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """
    contador = 0
    cap = cv.VideoCapture(src)
    new_frame, frame = cap.read()
    
    while new_frame:
        frame = en_paralelo(frame, 8)
        cv.imshow('Multiprocessing', frame)
        cv.moveWindow("Multiprocessing", 0, 0)
        contador += 1
        new_frame, frame = cap.read()

        if cv.waitKey(1) == ord("q"):
            new_frame = False

    return contador

def parallel_io(src):
    """
    Captura y muestra de frames con la clase QueueRead y ThreadedWindow. 
    La aplicación del efecto se hace en paralelo. 

    Args:
        - src (Filename || Index): nombre del archivo a recorrer o índice de cámara a activar. 

    Returns:
        - Open CV window: ventana con muestra de cada frame. 
        - Contador: cantidad de frames que recorre. 
    """
    read_obj = QueueRead(src)
    read_obj.start()
    time.sleep(0.1)
    frame = read_obj.read()
    frame = en_paralelo(frame, 8)
    show_obj = ThreadedWindow(frame)
    show_obj.start()
    contador = 1 

    while True:
        if(read_obj.more()):
            frame = en_paralelo(read_obj.read(), 8)
            show_obj.update(frame)
            contador += 1
        else:
            read_obj.stop()
            show_obj.stop()
            break
        if (not read_obj.enabled) or show_obj.stopped:
            read_obj.stop()
            show_obj.stop()
            break
    return contador

# INICIALIZACIÓN, PARSEO DE PARÁMETROS 
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="python tarea2.py ")
        
    parser.add_argument('-i', type=str, default="./video_boat.mp4") 

    parser.add_argument('--s', action='store_true', help='Serial processing')
    parser.add_argument('--tw', action='store_true', help='Threaded-Window test')
    parser.add_argument('--tr', action='store_true', help='Threaded-Read test')
    parser.add_argument('--qr', action='store_true', help='Queue-Read test')
    parser.add_argument('--tio', action='store_true', help='Threaded IO test')
    parser.add_argument('--tio2', action='store_true', help='Threaded IO test with no queue')
    parser.add_argument('--mp', action='store_true', help='Multiprocessing')
    parser.add_argument('--mpio', action='store_true', help='Multiprocessing with Threaded IO')
    args = parser.parse_args()
    
    start = datetime.now()
    if args.s:
        frames = serial(args.i)
    elif args.tr:
        frames = threaded_read(args.i)
    elif args.tw:
        frames = threaded_window(args.i)
    elif args.qr:
        frames = queue_read(args.i)
    elif args.tio:
        frames = threaded(args.i)
    elif args.tio2:
        frames = threaded2(args.i)
    elif args.mp:
        frames = parallel(args.i)
    elif args.mpio:
        frames = parallel_io(args.i)
    seconds = (datetime.now() - start).total_seconds()
    print("Frames {0} Time {1} FPS {2:.2f}".format(frames, seconds, round(float(frames)/float(seconds),6)))