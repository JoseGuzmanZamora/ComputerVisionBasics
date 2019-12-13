#!/usr/bin/env python
# coding: utf-8

# In[1]:


#José Alejandro Guzmán Zamora


# In[2]:


#José Alejandro Guzmán Zamora
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import plot 
import random



# In[3]:


def imgpad(img, r): 
    new_size = len(img[0]) + (2 * r)
    arreglo = np.zeros(( 2 * r, new_size), dtype=int)
    
    for i in range(len(img)):
        temporal = img[i]
        for j in range(r):
            temporal = np.insert(temporal, j, 0)
            temporal = np.insert(temporal, len(temporal), 0)
            
        arreglo = np.insert(arreglo, r + i, temporal, axis=0)
        
    return arreglo.astype(np.uint8)


# In[4]:


#disjoint set tree 

class nodo:
    p = None
    rank = None 
    
def make_set(x):
    x.p = x 
    x.rank = 0
    
def link(x, y):
    if x.rank > y.rank:
        y.p = x 
    else:
        x.p = y 
        if x.rank == y.rank:
            y.rank += 1

def find_set(x):
    if x != x.p:
        x.p = find_set(x.p)
    return x.p

def union(x , y):
    link(find_set(x), find_set(y))
    
def otro_union(x , y):
    x_root = find_set(x)
    y_root = find_set(y)
    
    if x_root == y_root:
        return 
    if x_root.rank < y_root.rank:
        x_root, y_root = y_root, x_root
        
    y_root.p = x_root 
    if x_root.rank == y_root.rank:
        x_root.rank += 1


# In[5]:


def vecindario(ubicacion):
    vecinos = []
    for i in range(3):
        vecinos.append([ubicacion[0] - 1, ubicacion[1] - 1 + i])
        vecinos.append([ubicacion[0] + 1, ubicacion[1] - 1 + i])
    vecinos.append([ubicacion[0], ubicacion[1] - 1])
    vecinos.append([ubicacion[0], ubicacion[1] + 1]) 
    return vecinos


# In[70]:


def connected_c(img):
    nodos = np.empty(0, dtype=object)
    vacia = np.zeros((len(img),len(img[0])), dtype=int).astype(np.uint8)
    label = 1
    for i in range(1, len(img)):
        for j in range(1, len(img[i])):
            if img[i,j] != 0:
                posibles_vecinos = vecindario([i,j])
                vecinos_verdaderos = []
                for k in range(len(posibles_vecinos)):
                    if vacia[posibles_vecinos[k][0], posibles_vecinos[k][1]] != 0:
                        vecinos_verdaderos.append(vacia[posibles_vecinos[k][0], posibles_vecinos[k][1]])
                #a este punto ya termino de ver si hay vecinos 
                if len(vecinos_verdaderos) > 0:
                    #cambiar el valor del label 
                    vacia[i, j] = min(vecinos_verdaderos)
                    #hacer union de equivalencias 
                    for z in range(len(vecinos_verdaderos)):
                        for t in range(len(vecinos_verdaderos)):
                            otro_union(nodos[vecinos_verdaderos[t] - 1], nodos[vecinos_verdaderos[z] - 1])
                        
                    
                else:
                    temporal = nodo()
                    make_set(temporal)
                    nodos = np.insert(nodos, label - 1, temporal)
                    vacia[i, j] = label 
                    label += 1 
                 
    #poner los labels de manera ordenada 
    labels = []
    for i in range(len(nodos)):
        labels.append([])
        for j in range(len(nodos)):
            if find_set(nodos[i]) == find_set(nodos[j]):
                labels[i].append(j + 1)
                
    print(vacia, '\n')
    
    #second pass 
    for r in range(1, len(vacia)):
        for u in range(1, len(vacia[r])):
            valor = vacia[r, u]
            if valor != 0:
                vacia[r, u] = min(labels[valor - 1]) 
                #vacia[r, u] = np.where(nodos == find_set(nodos[valor - 1]))[0] + 1
                
                
    print(vacia)
    return vacia               


# In[71]:


def labelview(labels):
    if len(labels.shape) == 2:
        labels = cv.cvtColor(labels, cv.COLOR_GRAY2RGB)
    
    colores = [[],[]]
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            valor = labels[i, j][0]
            if valor != 0:
                if valor in colores[0]:
                    indice = colores[0].index(valor)
                    color = colores[1][indice]
                else:
                    color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                    colores[0].append(valor)
                    colores[1].append(color)
                labels[i,j] = color 
    plot.imgview(labels)
        


# In[82]:


huella = cv.imread('./intentar_wiki.pgm', cv.IMREAD_GRAYSCALE)
plot.imgview(huella)


# In[83]:


import sys
np.set_printoptions(threshold=sys.maxsize)


# In[118]:


img = cv.imread('./fprint3.pgm', cv.IMREAD_GRAYSCALE)
img = img[300:500,300:500]
plot.imgview(img)
print(len(img))
print(len(img[0]))


# In[119]:


ret,img = cv.threshold(img,100,255,cv.THRESH_BINARY_INV)
plot.imgview(img)


# In[120]:


img = imgpad(img, 1)
img = connected_c(img)
labelview(img)


# In[ ]:




