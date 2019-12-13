#!/usr/bin/env python
# coding: utf-8

# In[1]:


# José Alejandro Guzmán Zamora


# In[2]:


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import plot 
import sys

get_ipython().run_line_magic('matplotlib', 'inline')


# In[128]:


def pad(img, r, v):
    size_x = len(img[0]) + (2 * r)
    size_y = len(img) + (2 * r)
    new = np.zeros((size_y, size_x))
    new[r:size_y - r, r:size_x - r] = img[0:len(img), 0:len(img[0])]
    
    if v == -1:
        new[0:r, r:size_x-r] = img[0,0:len(img[0])]
        new[size_y - r: size_y, r:size_x-r] = img[len(img) - 1,0:len(img[0])]
        new[0:size_y, 0:r] = new[0:size_y, r:r+1]
        new[0:size_y, size_x - r:size_x] = new[0:size_y, size_x - r - 1:size_x - r]
        
    else:
        new[0:r, 0:len(new[0])] = v
        new[r:size_y, 0:r] = v
        new[r:size_y, len(img[0]) + r:size_x] = v
        new[len(img) + r:size_y, r:size_x - r] = v
    return new 


# In[129]:


def convolve(img, kernel):
    assert len(img.shape) == 2
    assert kernel.shape[0] == kernel.shape[1]
    con_pad = pad(img,1,-1)
    nueva = np.zeros((len(con_pad),len(con_pad[0])))
    for i in range(1, len(con_pad) - 1):
        for j in range(1, len(con_pad[0]) - 1):
            vecindario = con_pad[i-1:i+2, j-1:j+2]
            nueva[i,j] = np.sum(np.multiply(kernel, vecindario))
    return(nueva[1:len(con_pad) - 1, 1:len(con_pad[0]) - 1])


# In[130]:


def float64_to_uint8(img):
    return plot.imgnorm(img)


# In[131]:


def gradient2magnitude(gx, gy):
    return np.sqrt(np.square(gx) + np.square(gy))


# In[156]:


def gradient2angle(gx, gy):
    #gx[gx == 0] = 0.1
    return np.arctan2(gx, gy)


# In[218]:


argumentos = sys.argv


path = './' + str(argumentos[1])
print(path)
nuevo = str(argumentos[2])

img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# In[166]:


gx = np.array([[-1, 0, 1] ,
                  [-2, 0, 2] ,
                  [-1, 0, 1]] , np.float64)
gy = np.array([[-1, -2, -1] ,
                  [0, 0, 0] ,
                  [1, 2, 1]] , np.float64)


# In[167]:


r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]


# In[168]:


r_x = convolve(r, gx)
r_y = convolve(r, gy)
g_x = convolve(g, gx)
g_y = convolve(g, gy)
b_x = convolve(b, gx)
b_y = convolve(b, gy)


# In[169]:


mag1 = float64_to_uint8(gradient2magnitude(r_x, r_y))
mag2 = float64_to_uint8(gradient2magnitude(g_x, g_y))
mag3 = float64_to_uint8(gradient2magnitude(b_x, b_y))


# In[170]:


final = cv.merge((mag1, mag2, mag3))
plot.imgview(final, None, None, "rgb_magnitudes")


# In[216]:


ang1 = float64_to_uint8(gradient2angle(r_x, r_y))
ang2 = float64_to_uint8(gradient2angle(g_x, g_y))
ang3 = float64_to_uint8(gradient2angle(b_x, b_y))


# In[217]:


final = cv.merge((ang1, ang2, ang3))
plot.imgview(final, None, None, "rgb_angles")

