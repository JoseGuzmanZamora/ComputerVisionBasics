#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2 as cv  
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


im = cv.imread(r'C:\Users\Chino Guzman\Desktop\lena.png', cv.IMREAD_COLOR)


# In[26]:


im.shape


# In[27]:


type(im)


# In[28]:


img = cv.cvtColor(im, cv.COLOR_BGR2RGB)


# In[29]:


fig = plt.figure(figsize=(10,10))
plt.imshow(img)


# In[ ]:




