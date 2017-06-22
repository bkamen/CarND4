import cv2
import numpy as np
import matplotlib.pyplot as plt

file = './test_images/test5.jpg'

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


img = cv2.imread(file)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
#img = clahe.apply(img)

img = adjust_gamma(img, .1)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = clahe.apply(img)

cv2.imshow('image', img)
cv2.waitKey(0)

#plt.imshow(adjust_gamma(img))
#plt.show()


