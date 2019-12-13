import cv2 as cv
import video

img = cv.imread('./cameraman_face.jpg')

for (i, resized) in enumerate(video.pyramid(img)):
    mensaje = "Layer " + str(i + 1)
    cv.imshow(mensaje,resized)
    cv.waitKey(0)
    print(resized.shape)
