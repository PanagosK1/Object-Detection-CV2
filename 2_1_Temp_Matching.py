from matplotlib import pyplot as plt
import cv2
import numpy as np
import random

'''
Function which change each pixel of the images, affected by the probability 
'''
def noise(image,prob):
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
                output[i][j] = image[i][j] + prob*random.uniform(0, 1)*image[i][j] - prob*random.uniform(0, 1)*image[i][j]
    return output

#load the image to check
img_rgb = cv2.imread('E:/PyCharm/PRLM_Lab_47138/Images to use/mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
noisy_img_0pc = noise(img_gray,0)
noisy_img_5pc = noise(img_gray,0.05)
noisy_img_10pc = noise(img_gray,0.1)
noisy_img_15pc = noise(img_gray,0.15)
noisy_img_20pc = noise(img_gray,0.2)


#load the template image we look for
template = cv2.imread('E:/PyCharm/PRLM_Lab_47138/Images to use/mario_coins_temp.jpg',0)
w, h = template.shape[::-1]

#run the templae matching
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

res_2 = cv2.matchTemplate(noisy_img_5pc,template,cv2.TM_CCOEFF_NORMED)
loc_2 = np.where(res_2 >= threshold)

res_3 = cv2.matchTemplate(noisy_img_10pc,template,cv2.TM_CCOEFF_NORMED)
loc_3 = np.where(res_3 >= threshold)

res_4 = cv2.matchTemplate(noisy_img_15pc,template,cv2.TM_CCOEFF_NORMED)
loc_4 = np.where(res_4 >= threshold)

res_5 = cv2.matchTemplate(noisy_img_20pc,template,cv2.TM_CCOEFF_NORMED)
loc_5 = np.where(res_5 >= threshold)


#mark the corresponding location(s)
for pt in zip(*loc[::-1]):
    cv2.rectangle(noisy_img_0pc, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

for pt in zip(*loc_2[::-1]):
    cv2.rectangle(noisy_img_5pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(noisy_img_10pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(noisy_img_15pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(noisy_img_20pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

#Here we plot the segmented images compared to each technique
plt.subplot(1,5,1), plt.imshow(noisy_img_0pc,'gray')
plt.title('Original Image with no noise')
plt.subplot(1,5,2), plt.imshow(noisy_img_5pc,'gray')
plt.title('Noisy Image with 5% noise')
plt.subplot(1,5,3), plt.imshow(noisy_img_10pc,'gray')
plt.title('Noisy Image with 10% noise')
plt.subplot(1,5,4), plt.imshow(noisy_img_15pc,'gray')
plt.title('Noisy Image with 15% noise')
plt.subplot(1,5,5), plt.imshow(noisy_img_20pc,'gray')
plt.title('Noisy Image with 20% noise')
plt.show()

#Erwthma B
#Xrhsh Gaussian filter

gaussian_img_0pc = cv2.GaussianBlur(noisy_img_0pc,(5,5),0)
gaussian_img_5pc = cv2.GaussianBlur(noisy_img_5pc,(5,5),0)
gaussian_img_10pc = cv2.GaussianBlur(noisy_img_10pc,(5,5),0)
gaussian_img_15pc = cv2.GaussianBlur(noisy_img_15pc,(5,5),0)
gaussian_img_20pc = cv2.GaussianBlur(noisy_img_20pc,(5,5),0)

#run the templae matching
res = cv2.matchTemplate(gaussian_img_0pc,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

res_2 = cv2.matchTemplate(gaussian_img_5pc,template,cv2.TM_CCOEFF_NORMED)
loc_2 = np.where(res_2 >= threshold)

res_3 = cv2.matchTemplate(gaussian_img_10pc,template,cv2.TM_CCOEFF_NORMED)
loc_3 = np.where(res_3 >= threshold)

res_4 = cv2.matchTemplate(gaussian_img_15pc,template,cv2.TM_CCOEFF_NORMED)
loc_4 = np.where(res_4 >= threshold)

res_5 = cv2.matchTemplate(gaussian_img_20pc,template,cv2.TM_CCOEFF_NORMED)
loc_5 = np.where(res_5 >= threshold)


#mark the corresponding location(s)
for pt in zip(*loc[::-1]):
    cv2.rectangle(gaussian_img_0pc, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

for pt in zip(*loc_2[::-1]):
    cv2.rectangle(gaussian_img_5pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(gaussian_img_10pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(gaussian_img_15pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

for pt in zip(*loc[::-1]):
    cv2.rectangle(gaussian_img_20pc, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


#Here we plot the segmented images compared to each technique
plt.subplot(1,5,1), plt.imshow(gaussian_img_0pc,'gray')
plt.title('Filtered Image with no noise')
plt.subplot(1,5,2), plt.imshow(gaussian_img_5pc,'gray')
plt.title('Filtered Image with 5% noise')
plt.subplot(1,5,3), plt.imshow(gaussian_img_10pc,'gray')
plt.title('Filtered Image with 10% noise')
plt.subplot(1,5,4), plt.imshow(gaussian_img_15pc,'gray')
plt.title('Filtered Image with 15% noise')
plt.subplot(1,5,5), plt.imshow(gaussian_img_20pc,'gray')
plt.title('Filtered Image with 20% noise')
plt.show()

