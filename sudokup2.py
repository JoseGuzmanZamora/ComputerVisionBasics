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
    Encuentra la figura de mayor área aproximadamente cuadrada dentro de la imagen. 
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
    perimetros = []
    for i in range(len(contours)):
        valor = cv.contourArea(contours[i])
        peri = cv.arcLength(contours[i],True)
        areas.append(valor)
        indices.append(i)
        perimetros.append(peri)
    information = sorted(zip(areas, indices, perimetros), reverse=True)

    dibujo = False
    #verificacion de que sea un cuadrado
    for i in range(0,15):
        if(i < len(information)):
            epsilon = 0.05*cv.arcLength(contours[information[i][1]],True)
            approx = cv.approxPolyDP(contours[information[i][1]],epsilon,True)
            if(len(approx) == 4):
                cv.drawContours(copia,[approx], -1, (0,255,0),4)
                dibujo = True
                break
        else:
            break

    approximacion = np.zeros((4,2), dtype="float32")
    approximacion[:] = approx[:,0]
    return [copia, approximacion] 

def specific_square(img, corners):
    """
    Aplica una transformacion lineal a una imagen en base a 4 esquinas. 
    
    Args:
        img(numpy array): imagen a procesar 
        corners(numpy array): cuatro esquinas que representan el destino
    Returns:
        warp(numpy array): imagen que representa el interior de las esquinas en su totalidad
    """
    #asegurarnos de que las esquinas esten en el mismo orden
    res = np.sum(corners, axis=1)
    index1 = np.where(res==max(res))
    index2 = np.where(res==min(res))
    #la sumatoria mayor nos dio la esquina inferior derecha, la mas pequeña la superior izquierda
    br = corners[index1][0]
    tl = corners[index2][0]
    #las elimino y despues ya solo tengo que comparar un eje para saber cual esquina es cual
    new = np.delete(corners,[index1,index2],0)
    index3 = np.where(new[:,0]==min(new[:,0]))
    index4 = np.where(new[:,0]==max(new[:,0]))
    tr = new[index3][0]
    bl = new[index4][0]
    origen = np.float32([tl, tr, br, bl])

    #aplicacion de get perspective y warp 
    filas, columnas = (500,500)
    objetivo = np.float32([[0,0],[0,columnas], [filas, columnas], [filas, 0]])
    matrix = cv.getPerspectiveTransform(origen, objetivo)
    warp = cv.warpPerspective(img, matrix, (filas,columnas))
    return warp

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
            result = search_square(imagen)
            mostrar(result[0])
            warped = specific_square(imagen,result[1])
            mostrar(warped)

