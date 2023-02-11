import cv2
import numpy as np

# getting image
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

# showing original image
cv2.imshow('original', image)

# horizontal edge detector
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
verti_convolving = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('horizontal edges', verti_convolving)

# vertical edge detector
kernel = np.array([[1,  1,  1],
                   [0,  0,  0],
                   [-1, -1, -1]])
horizontal_convolving = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('vertical edges', horizontal_convolving)

# 2d convolution filter
kernel = np.array([[1,  1,  1],
                   [0,  0,  0],
                   [-1, -1, -1]])
convolving_2D = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
cv2.imshow('2d convolution', convolving_2D)


# wait and quit
cv2.waitKey(0)
cv2.destroyAllWindows()