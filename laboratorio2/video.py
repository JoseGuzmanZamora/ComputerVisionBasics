import cv2 as cv 
import time


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

