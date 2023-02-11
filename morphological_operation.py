import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/Mihir/Downloads/smarties.png', cv2.IMREAD_GRAYSCALE)
_,mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)


kernal = np.ones((2,2), np.uint8)
dilation = cv2.dilate(mask, kernal, iterations=4)
erosion = cv2.erode(mask, kernal, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)



titles = ['image', 'mask', 'dialation', 'erosion', 'opening']
images = [img, mask, dilation, erosion, opening]
for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()