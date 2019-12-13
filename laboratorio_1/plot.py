#!/usr/bin/env python
# coding: utf-8

# In[12]:


#José Alejandro Guzmán Zamora
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# In[111]:


def imgview(img, ax=None, title=None, filename=None):
    """
    Mostrar Imagen. 
    
    Args:
        img (numpy.ndarray): Array con la información de la imagen
        title (String): String para asignar un título a la imagen, puede ser None si no se desea título
        filename (String): nombre del archivo para guardar visualización, no debe poner .png 
    
    Returns:
        Visualización (Matplotlib Plt): La imagen a color o en blanco y negro
        Visualización con título: en caso se provea valor de title
        Escritura de Visualización (.png file): en caso se provea valor de filename
    
    """ 
    
    if(len(img.shape) == 3):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        #plt.axis("off")
        plt.title(title, fontsize=18)
        plt.imshow(img)
    elif(len(img.shape) == 2):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        #plt.axis("off")
        plt.title(title, fontsize=18)
        plt.imshow(img, vmin=0, vmax=255)
        
    
    if(filename != None):
        plt.savefig(filename + ".png")
        
    if(ax != None):
        if(ax):
            if(len(img) <= 8):
                valor = 1
            else:
                valor = len(img) / 8
            ax1 = plt.gca()
            ax1.set_xticks(np.arange(-.5, len(img[0]), valor))
            ax1.set_yticks(np.arange(-.5, len(img), valor))
            ax1.set_xticklabels(np.arange(0, len(img[0]), valor))
            ax1.set_yticklabels(np.arange(0, len(img), valor))
            plt.axis("on")
            plt.grid(color='w', linestyle='-', linewidth=2)
        else:
            plt.axis("off")
    else:
        plt.axis("off")
    plt.show()


# In[109]:


def imgcmp(img1, img2, ax=None, title1=None, title2=None, filename=None):
    """
    Comparación de 2 Imágenes. 
    
    Args:
        img1 (numpy.ndarray): Array con la información de la primera imagen
        img2 (numpy.ndarray): Array con la información de la segunda imagen
        title1 (String): String para asignar un título a la primera imagen, puede ser None si no se desea título
        title2 (String): String para asignar un título a la segunda imagen, puede ser None si no se desea título
        filename (String): nombre del archivo para guardar visualización, no debe poner .png
    
    Returns:
        Visualización (Matplotlib Plt): Las imágenes una a la par de la otra
        Visualización con títulos: en caso se provea valor de title correspondiente
        Escritura de Visualización (.png file): en caso se provea valor de filename
    
    """
    
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1,2, figsize=(15,15))
    
    #axes[0].axis("off")
    axes[0].set_title(title1, fontsize=18)
    axes[0].imshow(img1, vmin=0, vmax=255)
    
    #axes[1].axis("off")
    axes[1].set_title(title2, fontsize=18)
    axes[1].imshow(img2, vmin=0, vmax=255)
    
    if(filename != None):
        plt.savefig(filename + "png")
        
    
    if(ax != None):
        if(ax):
            if(len(img1) <= 8):
                valor = 1
            else:
                valor = len(img1) / 8
            
            if(len(img2) <= 8):
                valor2 = 1
            else:
                valor2 = len(img2) / 8
                
            axes[0].set_xticks(np.arange(-.5, len(img1[0]), valor))
            axes[0].set_yticks(np.arange(-.5, len(img1), valor))
            axes[0].set_xticklabels(np.arange(0, len(img1[0]), valor))
            axes[0].set_yticklabels(np.arange(0, len(img1), valor))
            axes[1].set_xticks(np.arange(-.5, len(img2[0]), valor2))
            axes[1].set_yticks(np.arange(-.5, len(img2), valor2))
            axes[1].set_xticklabels(np.arange(0, len(img2[0]), valor2))
            axes[1].set_yticklabels(np.arange(0, len(img2), valor2))
            axes[0].axis("on")
            axes[1].axis("on")
            axes[0].grid(color='w', linestyle='-', linewidth=2)
            axes[1].grid(color='w', linestyle='-', linewidth=2)
        else:
            axes[0].axis("off")
            axes[1].axis("off")
    else:
        axes[0].axis("off")
        axes[1].axis("off")
        
    plt.show()


# In[3]:


def split_rgb(img, filename = None):
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(221)
    ax1.imshow(img)
    ax1.set_title("RGB")
    plt.axis("off")
    ax2 = fig.add_subplot(222)
    ax2.imshow(img[:,:,0], cmap='gray', vmin=0, vmax=255)
    ax2.set_title("R")
    plt.axis("off")
    ax3 = fig.add_subplot(223)
    ax3.imshow(img[:,:,1], cmap='gray', vmin=0, vmax=255)
    ax3.set_title("G")
    plt.axis("off")
    ax4 = fig.add_subplot(224)
    ax4.imshow(img[:,:,2], cmap='gray', vmin=0, vmax=255)
    ax4.set_title("B")
    plt.axis("off")
    
    if filename != None:
        plt.savefig(filename)
    plt.show()


# In[4]:


def hist(img, filename=None):
    fig = plt.figure(figsize=(30,10))
    ax1 = fig.add_subplot(111)
    
    colors = ['r', 'g', 'b']
    
    if(len(img.shape) == 3):
        for i,color in enumerate(colors):
            histr = cv.calcHist([img], [i], None, [256], [0,256])
            ax1.plot(histr, c=color)
    else:
        for i,color in enumerate(colors):
            histr = cv.calcHist([img], [0], None, [256], [0,256])
            ax1.plot(histr, c=color)
    plt.show()
    return histr


# In[5]:


def imgnorm(img):
    vmin, vmax = img.min(), img.max()
    normalized = []
    delta = vmax - vmin 
    
    for p in img.ravel():
        normalized.append(255*((p-vmin)/delta))
    img_normalized = np.array(normalized).astype(np.uint8).reshape(img.shape)
    return img_normalized


# In[6]:


def cdf(hist, size):
    new = []
    pasado = 0
    for i in range(0, len(hist)):
        pasado += hist[i]
        new.append(pasado / size)
        
    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.plot(hist)
    
    ax2 = ax1.twinx()
    ax2.plot(new, color="black")
    
    plt.show()
    return new


# In[7]:


def equalizar(imgg):
    normalizada = imgnorm(imgg)
    histograma = hist(normalizada)
    cdf_par = cdf(histograma, normalizada.size)
    cdf_e = np.ma.masked_equal(cdf_par, 0)
    cdf_e = (cdf_e - cdf_e.min()) * 255 / (cdf_e.max() - cdf_e.min())
    cdff = np.ma.filled(cdf_e,0).astype('uint8')
    resultado = cdff[normalizada].reshape(normalizada.shape)
    return resultado

