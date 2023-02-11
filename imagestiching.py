import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image= cv.imread('lena.jpg',1)
cv.imshow('original', image)
rows,cols,ht=image.shape
hist1= cv.calcHist([image],[0],None,[256],[0,256])
cv.normalize(hist1,hist1,0,255,cv.NORM_MINMAX)
plt.plot(hist1)
##rotation method
arrayMatrix= cv.getRotationMatrix2D((0,0),10,1)
print(arrayMatrix)
new_image= cv.warpAffine(image,arrayMatrix,(cols,rows))
cv.imshow('output',new_image)


hist2= cv.calcHist([new_image],[0],None,[256],[0,256])
print("this is the value obtained by comparing original image and image after rotating")
metric_val_cmp_rotational_original= cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)
print(metric_val_cmp_rotational_original)
cv.normalize(hist2,hist2,0,255,cv.NORM_MINMAX)

plt.plot(hist2)

cmp_rotational_ori = plt.show()



##Translation method

#      |1 0 Tx|
#      |0 1 Ty|

height, width = image.shape[:2]
print(height,width)

height_fourth,width_fourth= height/4, width/4

translationMatrix = np.float32([[1,0,height_fourth],[0,1,width_fourth]])
print(translationMatrix)


translationMethod = cv.warpAffine(image, translationMatrix,(width,height))

cv.imshow('translation',translationMethod)
hist3= cv.calcHist([new_image],[0],None,[256],[0,256])
cv.normalize(hist3,hist3,0,255,cv.NORM_MINMAX)
print("this is the value obtained by comparing original image and iamge after applying translation method")
metric_val_cmp_translation_original= cv.compareHist(hist1,hist3,cv.HISTCMP_CORREL)
print(metric_val_cmp_translation_original)
plt.plot(hist1)
plt.plot(hist3)

cmp_translation_ori= plt.show()

Mask_scale = np.float32([
    [3, 0, 0],
    [0, 2, 0]]

)
scalling_image = cv.warpAffine(image, Mask_scale, (rows*2, cols*2))
cv.imshow('scaling',scalling_image)
hist4= cv.calcHist([new_image],[0],None,[256],[0,256])
cv.normalize(hist4,hist4,0,255,cv.NORM_MINMAX)
print("this is the value obtained by comparing original image and iamge after applying translation method")
metric_val_cmp_translation_original= cv.compareHist(hist1,hist3,cv.HISTCMP_CORREL)
print(metric_val_cmp_translation_original)
plt.plot(hist1)
plt.plot(hist4)

plt.show()


cv.waitKey(0)
cv.destroyAllWindows()