#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import plot 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


im = cv.threshold(cv.imread('./cameraman1.png', cv.IMREAD_GRAYSCALE), 100, 255, cv.THRESH_BINARY)[1]


# In[ ]:


sobelx = cv.Sobel((im, cv.CV_64F), 1, 0, ksize=5)
sobelx = cv.Sobel((im, cv.CV_64F), 0, 1, ksize=5)


# convertir gradiente a unsigned int 
# se puede hacer con valor absoluto si no importa la simetria 
# tambien se puede utilizar normalizacion 
# 
# laplacian cmp edges  
# 63
# 
