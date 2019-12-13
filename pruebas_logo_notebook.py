#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 
import plot

get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


img = cv.imread('./ufmlogo.png',0)
#img = cv.blur(img, (8, 8))
#ret,img = cv.threshold(img,100,255,cv.THRESH_BINARY)
img = cv.Canny(img, 50, 200)
plot.imgview(img)


# In[4]:


img2 = cv.imread('./encontrar.png', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('./encontrar.png')
img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
plot.imgview(img3)


# In[5]:


template = img2[250:450, 420:600]
plot.imgview(template)


# In[57]:


w, h = template.shape[::-1]
res = cv.matchTemplate(img2,template,cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)


# In[58]:


top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img3,top_left, bottom_right, 255, 2)
plot.imgview(img3)


# In[59]:


#un poco mas formal 
#esto muestra el ancho y alto del template, hace -1 porque shape muestra alto y ancho 
w, h = template.shape[::-1]

#todos los metodos que hay, seria de probar cada uno 
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


# In[66]:


for meth in methods:
    #esto es el numero de meth 
    method = eval(meth)
    
    #aplicar template matching 
    res = cv.matchTemplate(img2,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
    #min value es lo menos que saco la tranformacion 
    #max value es lo que mas saco la transormacion 
    #min loc creo que es el punto en el que encontro eso y max loc tambien 
    
    #los ultimos dos metodos sacan diferente el top left corner 
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    #con esta operacion calcula cual es la ubicacion el bottom right corner para hacer el rectangulo 
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    plot.imgview(img3)


# In[61]:


get_ipython().run_cell_magic('time', '', "#siguiente prueba\n\nimg3 = cv.imread('./encontrar.png')\nimg3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)\nh,w = template.shape\nres = cv.matchTemplate(img2,template,cv.TM_CCOEFF)\nmin_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)\ntop_left = max_loc\nbottom_right = (top_left[0] + w, top_left[1] + h)\ncv.rectangle(img3,top_left, bottom_right, (0,255,0), 3)\nplot.imgview(img3)")


# In[65]:


print(img.shape)


# In[ ]:




