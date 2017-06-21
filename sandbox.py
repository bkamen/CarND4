import cv2
import numpy as np
import matplotlib.pyplot as plt

file = './test_images/straight_lines1.jpg'

img = cv2.imread(file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
img = clahe.apply(img)

plt.imshow(img, cmap='gray')
plt.show()

