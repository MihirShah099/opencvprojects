import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/Users/Mihir/Downloads/lena.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
