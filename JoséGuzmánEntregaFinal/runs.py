import os 
import cv2 as cv 
import numpy as np


'''for filename in os.listdir("./"):
    if "ttf" in filename:
        print(filename)
        os.system("python font2img.py --f ./" + filename + " -p")'''
# ya se corrio el script

valores_y = []
valores_x = []

contador = 0
for filenames in os.listdir("./digits_synthetic"):
        for filename in os.listdir("./digits_synthetic/" + filenames):
                path = "./digits_synthetic/" + filenames + "/" + filename
                img = cv.imread(path, 0)
                img = cv.bitwise_not(img)
                img = cv.resize(img,(28,28))
                valores_x.append(img)
                valores_y.append(contador)
        contador += 1

verificarx = np.asarray(valores_x)
verificary = np.asarray(valores_y)
todo = [verificarx,verificary]
np.save("xs",verificarx)
np.save("ys",verificary)
