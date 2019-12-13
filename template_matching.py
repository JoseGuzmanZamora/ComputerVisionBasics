#!/usr/bin/env python
# coding: utf-8

# # Template Matching

# In[6]:


#import sys ; sys.path.append("../") # osx
#import cvlib

import cv2 as cv
import plot
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


#PATH = '../im/'
img = cv.imread('./cameraman.png', cv.IMREAD_GRAYSCALE)
img2 = img.copy()
template = img[150:300, 300:500]
#template = cv.imread('./cameraman_face.jpg', cv.IMREAD_GRAYSCALE)

'''scale = 0.4
template = cv.resize(template, None,fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
w, h = template.shape[::-1]'''
print(img.shape)
print(template.shape)


# In[70]:


plot.imgcmp(img,template)


# In[71]:


methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv.matchTemplate(img, template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()


# In[76]:


def pyramid(img, scale=0.5, min_size=(32,32)):
    """
        Build a pyramid for an image until min_size
        dimensions are reached. 
        
        Args:
            img (numpy array): source image 
            scale(float): scaling factor
            min_size(tuple): minimum size of pyramid top level 
        Returs:
            Pyramid generator
    """
    
    yield img 
    
    while True:
        img = cv.resize(img, None,fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
        if (img.shape[0] < min_size[0]) and (img.shape[1] < min_size[1]):
            break
        yield img 
    


# In[77]:


for i in pyramid(img):
    cv.imgshow
    print(i.shape)


# In[ ]:




