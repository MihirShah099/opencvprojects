import cv2
import numpy as np
img = cv2.imread('C:/Users/Mihir/Downloads/messi5.jpg')
img1 = cv2.imread('C:/Users/Mihir/Downloads/opencvlogo.png')
print(img.shape)
print(img.size)
print(img.dtype)
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))
img =cv2.resize(img,(512,512))
img1 = cv2.resize(img1,(512,512))
ball = img[280:340,330:390]
img[273:333,100:160]=ball
#dis = cv2.add(img,img1,)
dis = cv2.addWeighted(img,.9,img1,.1,0)
cv2.imshow('image',dis)

cv2.waitKey(0)
cv2.destroyAllWindows()
