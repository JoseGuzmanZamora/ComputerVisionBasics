#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 
import plot

from mpl_toolkits.mplot3d import axes3d

plt.style.use('dark_background')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


img = cv.imread("./wikipedia.png", cv.IMREAD_GRAYSCALE)

img = plot.equalizar(img)
plot.imgview(img)


# In[4]:


thresh_val = 80
ret, thresh = cv.threshold(img, thresh_val, 255, cv.THRESH_BINARY)


# In[5]:


print(ret)


# In[6]:


plot.imgcmp(img, thresh)


# In[7]:


binarized = cv.threshold(img, thresh_val, 255, cv.THRESH_TRUNC)[1]
plot.imgcmp(img, binarized)


# In[ ]:


ret_otsu, otsu = cv.threshold(img, 0, 255, cv.)

