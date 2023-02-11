import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from my_imfilter import vis_hybrid_image, load_image, save_image
from student_code import my_imfilter, create_hybrid_image

image1 = load_image('D:/opencvprojects/data/4a_einstein.bmp')
image2 = load_image('D:/opencvprojects/data/4b_marilyn.bmp')


# display the dog and cat images
plt.figure(figsize=(3,3)); plt.imshow((image1*255).astype(np.uint8));
plt.figure(figsize=(3,3)); plt.imshow((image2*255).astype(np.uint8));


cutoff_frequency = 4
filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                               sigma=cutoff_frequency)
filter = np.dot(filter, filter.T)

# let's take a look at the filter!
plt.figure(figsize=(4,4)); plt.imshow(filter);


blurry_dog = my_imfilter(image1, filter)
plt.figure(); plt.imshow((blurry_dog*255).astype(np.uint8));

low_frequencies, high_frequencies, hybrid_image = create_hybrid_image(image1, image2, filter)
vis = vis_hybrid_image(hybrid_image)


plt.figure(); plt.imshow((low_frequencies*255).astype(np.uint8));
plt.figure(); plt.imshow(((high_frequencies+0.5)*255).astype(np.uint8));
plt.figure(figsize=(20, 20)); plt.imshow(vis);

save_image('D:/opencvprojects/results/low_frequencies.jpg', low_frequencies)
save_image('D:/opencvprojects/results/high_frequencies.jpg', high_frequencies+0.5)
save_image('D:/opencvprojects/results/hybrid_image.jpg', hybrid_image)
save_image('D:/opencvprojects/results/hybrid_image_scales.jpg', vis)