#!/usr/bin/env python
# coding: utf-8

# In[3]:


#José Alejandro Guzmán Zamora
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import plot 
import random
import sys


# In[4]:


def imgpad(img, r): 
    """
    Agregar 'pad' o borde a la imagen. 
    
    Args:
        img (Numpy nd Array): arreglo con la información de la imagen 
        r (int): cantidad de capas que se agregan por lado
    Returns:
        Imagen Nueva (Numpy nd Array): información de la imagen nueva con bordes agregados
    """
    new_size = len(img[0]) + (2 * r)
    arreglo = np.zeros(( 2 * r, new_size), dtype=int)
    
    for i in range(len(img)):
        temporal = img[i]
        for j in range(r):
            temporal = np.insert(temporal, j, 0)
            temporal = np.insert(temporal, len(temporal), 0)
            
        arreglo = np.insert(arreglo, r + i, temporal, axis=0)
        
    return arreglo.astype(np.uint8)


# In[5]:


#Disjoint Set Data Structure 
# https://medium.com/100-days-of-algorithms/day-41-union-find-d0027148376d
def find(data, i):
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]

def union(data, i, j):
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        data[pi] = pj
        
def connected(data, i, j):
    return find(data, i) == find(data, j)


# In[6]:


def labelview(labels, filename=None):
    """
    Visualización de labels asignados con colores. 
    
    Args:
        labels (Numpy nd Array): arreglo en formato de imagen con labels indicados
        filename (string): string con el nombre del archivo a guardar
    Returns:
        Visualización (plot.imgview): imagen con los colores asignados en lugar de las etiquetas
    """
    
    nueva = np.zeros((len(labels),len(labels[0])), dtype=int).astype(np.uint8)
    
    if len(nueva.shape) == 2:
        nueva = cv.cvtColor(nueva, cv.COLOR_GRAY2RGB)
    
    colores = [[],[]]
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            valor = labels[i, j]
            if valor != 0:
                if valor in colores[0]:
                    indice = colores[0].index(valor)
                    color = colores[1][indice]
                else:
                    color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                    colores[0].append(valor)
                    colores[1].append(color)
                nueva[i,j] = color 
    plot.imgview(nueva, None, None, filename)
        


# In[7]:


def intentarotravez(img):
    connections = []
    label = 1 
    vacia = np.zeros((len(img),len(img[0])), dtype=int).astype(np.int64)
    
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i,j] != 0:
                vecinos = vacia[i - 1:i + 2, j - 1:j + 2]

                if np.count_nonzero(vecinos) > 0:
                    count_zeros = np.nonzero(vecinos)
                    vecinos_verdaderos = []
                    for k in range(len(count_zeros[0])):
                        value = vecinos[count_zeros[0][k], count_zeros[1][k]]
                        if value not in vecinos_verdaderos:
                            vecinos_verdaderos.append(value)
                    vacia[i,j] = min(vecinos_verdaderos)
                    
                    vecinos_verdaderos.sort()
                    for m in range(len(vecinos_verdaderos)):
                        valor = [min(vecinos_verdaderos) - 1, vecinos_verdaderos[m] - 1]
                        if not(valor in connections):
                            connections.append(valor)
                        
                        
                else:
                    vacia[i,j] = label
                    label += 1   
                    
    info = [i for i in range(label - 1)]

    for i, j in connections:
        union(info, i, j)
    
    for i in range(len(vacia)):
        for j in range(len(vacia[i])):
            if vacia[i,j] != 0:
                vacia[i,j] = find(info, vacia[i,j] - 1)
    
    return vacia 


# In[10]:


argumentos = sys.argv


path = './' + str(argumentos[1])
print(path)
nuevo = str(argumentos[2])


img = cv.imread(path, cv.IMREAD_GRAYSCALE)
img = cv.bitwise_not(img)
siguiente = cv.GaussianBlur(img,(3,3),0)

kernel = np.ones((2,2),np.uint8)

ret,siguiente = cv.threshold(siguiente,127,255,cv.THRESH_BINARY)
siguiente = cv.erode(siguiente,kernel,iterations = 2)


siguiente = cv.morphologyEx(siguiente,cv.MORPH_OPEN,kernel, iterations = 1)
siguiente = cv.dilate(siguiente,kernel,iterations = 2)

labelview(intentarotravez(siguiente), nuevo)

