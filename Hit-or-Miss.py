#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 
import plot

get_ipython().run_line_magic('matplotlib', 'inline')


# # Hit-or-Miss
# 
# ## Transformación basada en una secuencia de operaciones morfológicas realizadas sobre una imagen binaria. 
# 
# ## Operación principal: Erosión 
# ## Utiliza 2 kernels, uno de hit y otro de miss 
# 
# ## Secuencia:
# - Erosión sobre la imagen inicial con el kernel de hit
# - Erosión sobre el complemento de la imagen inicial con el kernel de miss
# - Intersección de ambos resultados (AND)
# 
# 
# 
# 

# In[3]:


def hit_miss(img, kernel_hit, kernel_miss):
    img2 = cv.bitwise_not(img)
    erode_1 = cv.erode(img, kernel_hit, iterations=1)
    erode_2 = cv.erode(img2, kernel_miss, iterations=1)
    return cv.bitwise_and(erode_1, erode_2)


# In[4]:


imagen = cv.imread('./intentar.pgm', cv.IMREAD_GRAYSCALE)


# In[5]:


plot.imgview(imagen, True)


# In[6]:


kernel_hit = np.array([[0, 255, 0] ,
                  [0, 255, 0] ,
                  [0, 255, 0]] , np.uint8)
kernel_miss = np.array([[255,0,255],
                     [255,0,255],
                     [255,0,255]] , np.uint8)


# In[7]:


plot.imgcmp(kernel_hit, kernel_miss, True)


# In[8]:


plot.imgview(hit_miss(imagen, kernel_hit, kernel_miss), True)


# In[9]:


kernel_hit2 = np.array([[0, 255, 0] ,
                  [255, 0, 255] ,
                  [0, 255, 0]] , np.uint8)
kernel_miss2 = np.array([[0,0,0],
                     [0,255,0],
                     [0,0,0]] , np.uint8)


# In[10]:


plot.imgcmp(kernel_hit2, kernel_miss2, True)


# In[11]:


plot.imgview(hit_miss(imagen, kernel_hit2, kernel_miss2), True)


# In[12]:


kernel_hit3 = np.array([[255, 255, 0] ,
                  [0, 255, 0] ,
                  [0, 255, 0]] , np.uint8)
kernel_miss3 = np.array([[0,0,255],
                     [0,0,255],
                     [0,0,255]] , np.uint8)


# In[13]:


plot.imgcmp(kernel_hit3, kernel_miss3, True)


# In[14]:


plot.imgview(hit_miss(imagen, kernel_hit3, kernel_miss3), True)


# In[15]:


people = cv.imread('./edificios.png', cv.IMREAD_GRAYSCALE)


# In[16]:


plot.imgview(people,True, "Original", "original")


# In[17]:


binarizada = cv.adaptiveThreshold(people,255,cv.ADAPTIVE_THRESH_MEAN_C,            cv.THRESH_BINARY,11,2)
plot.imgview(binarizada, True, "Binarizada", "binarizada")


# In[18]:


kernel_hit4 = np.array([[0, 0, 0] ,
                  [255, 0, 0] ,
                  [255, 255, 0]] , np.uint8)
kernel_miss4 = np.array([[0,255,255],
                     [0,255,255],
                     [0,0,0]] , np.uint8)


# In[19]:


resultado = hit_miss(binarizada, kernel_hit4, kernel_miss4)


# In[20]:


plot.imgview(resultado)


# In[ ]:




