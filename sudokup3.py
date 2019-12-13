# José Alejandro Guzmán Zamora
import keras
import cv2 as cv 
import numpy as np
from keras.models import load_model

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
    filas, columnas = (450,450)
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

def find_grid(img):
    """
    Localiza el grid del sudoku para saber cuáles casillas tienen número. 
    
    Args:
        img(numpy array): imagen cuadrada del sudoku 

    Returns:
        refinado(numpy array): conjunto de imágenes binarias correspondientes 
        a cada número encontrado
        ubicaciones(numpy array): ubicación de cada imagen dentro del arreglo de 
        81 valores
    """

    grid = 9
    size = grid * grid
    x = img.shape[0]
    newx = round(x / 9)
    lastx = 600 - (newx * 8)
    
    img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ver = cv.adaptiveThreshold(img_bw,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    cv.imshow("binaria", ver)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    imagen_temporal = np.zeros((newx,newx), dtype='uint8')
    images = []
    contador_c = 0
    contador_f = 0
    for i in range(1,size + 1):
        sup_izq = [newx * contador_c, newx * contador_f]

        imagen_temporal = img_bw[sup_izq[0]:sup_izq[0] + newx, sup_izq[1]:sup_izq[1] + newx]

        images.append(imagen_temporal)
        imagen_temporal = np.zeros((newx,newx), dtype='uint8')
        contador_f += 1
        if i % grid == 0:
            #llega al final de las columnas
            contador_c += 1
            contador_f = 0
    #ahorita ya tengo todas las mini imagenes guardadas en el arreglo 

    grid_refinado = []
    ubicaciones = 0
    ubicacionesa = []
    #ahora voy a recorrerlo, aplicando ciertas transformaciones y guardandolas de nuevo
    for i in images:
        new_size = 250
        #crop image 
        reduccion = 0.15
        cantidad_r = round(new_size * reduccion)
        retornar = cv.resize(i,(new_size, new_size))
        imagen_mini = retornar[cantidad_r:new_size-cantidad_r,cantidad_r:new_size - cantidad_r]

        ret,binary = cv.threshold(imagen_mini,140,255,cv.THRESH_BINARY_INV)
        
        cropped_size = imagen_mini.shape[0]
        #empezar a ver si tiene pixeles blancos
        linea_media_h = binary[round(cropped_size / 2),:]
        linea_media_v = binary[:,round(cropped_size / 2)]
        
        blanco1 = 0
        blanco2 = 0
        for j in linea_media_h:
            if(j == 255):
                blanco1 += 1
        for j in linea_media_v:
            if(j == 255):
                blanco2 += 1
        valor_verificacion = (blanco1 + blanco2) / 2
        
        if valor_verificacion > 5 :
            grid_refinado.append(binary)
            ubicacionesa.append(ubicaciones)
        ubicaciones += 1
    return [grid_refinado,ubicacionesa]

def predict(images):
    """
    Utilización de CNN para identificar números 
    
    Args:
        images(numpy array): conjunto de imágenes con números a identificar

    Returns:
        respuesta(numpy array): arreglo con los números obtenidos del modelo 
    """
    modelo = load_model('modelo2.h5')

    arreglo = []
    for i in images:
        imagen = cv.resize(i, (28,28))
        imagen = imagen.reshape(28,28,1)
        arreglo.append(imagen)
    arreglo2 = np.asarray(arreglo)  

    resultado = modelo.predict(arreglo2)
    respuesta = np.argmax(np.round(resultado),axis=1)
    return respuesta

def make_grid(valores):
    gridf = np.zeros((81), dtype='int64')
    gridf[valores[1][:]] = valores[0][:]
    retornar = np.reshape(gridf,(9,9))
    return retornar

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
            grid_encontrado = find_grid(warped)
            numeros = predict(grid_encontrado[0])
            a_solucionar = make_grid([numeros,grid_encontrado[1]])
            print(a_solucionar)

