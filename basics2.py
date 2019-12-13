#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 
import plot

from mpl_toolkits.mplot3d import axes3d

get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


img = cv.imread('./imagen_wikipedia.pgm')
print(img)


# # Erosion

# In[3]:


plot.imgview(img)


# In[6]:


kernel_miss = np.array([[0, 255, 0] ,
                  [255, 0, 255] ,
                  [0, 255, 0]] , np.uint8)
kernel_hit = np.array([[0,0,0],
                     [0,255,0],
                     [0,0,0]] , np.uint8)


# In[8]:


plot.imgview(kernel_hit)


# In[9]:


erosion = cv.erode(img, kernel_miss, iterations=2)


# In[10]:


plot.imgview(erosion)


# In[11]:


thresh_val = 70
ret, thresh = cv.threshold(img, thresh_val, 255, cv.THRESH_BINARY)


# In[13]:


plot.imgview(thresh)


# In[25]:


erosion = cv.erode(thresh, kernel, iterations=20)


# DILATION, OPENING, CLOSING, MORFLOGICAL OPERATORS, DILATION EROSION.  

# In[26]:


binarized = cv.threshold(img, thresh_val, 255, cv.THRESH_TRUNC)[1]


# In[28]:


plot.imgview(thresh)


# In[4]:


img = cv.imread('C:/Users/joseg/OneDrive/Desktop/CVJN/intentar.pgm', cv.IMREAD_GRAYSCALE)


# In[5]:


plot.imgview(img)


# In[ ]:




