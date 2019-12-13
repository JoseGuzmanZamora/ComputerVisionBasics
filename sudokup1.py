# José Alejandro Guzmán Zamora

import cv2 as cv 
import numpy as np

def lectura(path):
    """
    Lectura de imagen utilizando función cv.imread() 
    """
    img = cv.imread(path)
    return img

def search_square(img):
    """
    Encuentra la figura de mayor área dentro de la imagen. 
    Aplica blur, canny, dilation y erosion antes de aplicar la función de cv.findContours(). 
    
    Args:
        img(numpy array): imagen a procesar 
    Returns:
        copia(numpy array): copa de imagen de entrada, con el contorno dibujado
    """
    copia = img.copy()

    #KERNELS
    kernel_g = np.ones((3,3), np.uint8)
    kernel_c = np.ones((2,2), np.uint8)

    #TRANSFORMACIONES PREVIAS 
    copia_img = img.copy()
    copia_img = cv.cvtColor(copia_img, cv.COLOR_BGR2GRAY)
    blur = cv.blur(copia_img,(3,3))
    edges = cv.Canny(blur, 50,150)

    dilation = cv.dilate(edges, kernel_g, iterations=3)
    continuar = cv.erode(dilation, kernel_c, iterations=1)

    #ENCONTRAR LOS CONTORNOS 
    contours,hierarchy = cv.findContours(continuar, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    areas = []
    indices = []
    for i in range(len(contours)):
        valor = cv.contourArea(contours[i])
        areas.append(valor)
        indices.append(i)
    information = sorted(zip(areas, indices), reverse=True)

    cv.drawContours(copia,[contours[information[0][1]]], -1, (0,255,0),2)
    return copia 

def mostrar(img):
    """
    Mostrar una imagen arbitraria. 
    Hace un reisze si el ancho o largo es mayor a 600, con la intención de siempre ver la imagen.
    """
    alto = img.shape[0]
    ancho = img.shape[1]
    if(alto > 600 or ancho > 600):
        mayor = max(alto, ancho)
        resize = round(mayor / 600) 
        nuevo_alto = round(alto/resize)
        nuevo_ancho = round(ancho/resize)
        img = cv.resize(img,(nuevo_ancho,nuevo_alto))
    cv.imshow('SUDOKU',img)
    cv.moveWindow('SUDOKU',0,0)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''      
contorno = contours[information[0][1]]
epsilon = 0.05*cv.arcLength(contorno,True)
approx = cv.approxPolyDP(contorno,epsilon,True)
'''

# INICIALIZACIÓN, PARSEO DE PARÁMETROS 
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="python solver.py ")
        
    parser.add_argument('--p', type=str, help="Path de la imagen a procesar", action="store")
    
    args = parser.parse_args()

    if(args.p):
        imagen = lectura(args.p)
        if imagen is None:
            print("Error: no se pudo leer imagen. Revise el path.")
        else:
            mostrar(search_square(imagen))

