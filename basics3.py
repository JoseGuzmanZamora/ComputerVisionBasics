#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 
import plot
import math 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


img = cv.imread("./wikipedia.png", cv.IMREAD_GRAYSCALE)


# In[16]:


print(img.shape)
plot.imgview(img)
normalizada = plot.imgnorm(img)


# In[17]:


hello = plot.hist(img)
print(normalizada.shape)


# In[18]:


plot.hist(normalizada)
plot.imgview(normalizada)


# In[19]:


histograma = plot.hist(normalizada)


# In[20]:


cddf = plot.cdf(histograma, normalizada.size)
print(normalizada.shape)


# In[21]:


equalizada = plot.equalizar(cddf, normalizada)


# In[22]:


plot.imgview(equalizada)


# In[12]:


print(equalizada.shape)
nuevo_histo = plot.hist(equalizada)


# In[13]:


print(plot.cdf(nuevo_histo, equalizada.size))


# In[ ]:




