import cv2
import numpy as np


# load the image to check
img_rgb = cv2.imread('E:/PyCharm/PRLM_Lab_47138/Images to use/mario.jpg')

# Convert to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# load the template image we look for
template = cv2.imread('E:/PyCharm/PRLM_Lab_47138/Images to use/mario_coins_temp.jpg', 0)

source_wid, source_hei = img_gray.shape[::-1]
temp_w, temp_h = template.shape[::-1]

temp_hist = cv2.calcHist([template], [0], None, [256], [0, 256])
temp_wid_2 = int(temp_w / 2)
temp_hei_2 = int(temp_h / 2)
thresh = 0.6
thresh_2 = 0.7

#clone the original image
img_clone = img_gray.copy

# Crop out the window and calculate the histogram
for r in range(temp_wid_2, source_wid - temp_wid_2, temp_wid_2):
    for c in range(temp_hei_2, source_hei - temp_hei_2, temp_hei_2):
        window = img_gray[r-temp_wid_2:r + temp_wid_2,c-temp_hei_2 :c + temp_hei_2]
        hist = cv2.calcHist([window], [0], None, [256], [0, 256])
        result = cv2.compareHist(hist, temp_hist, cv2.HISTCMP_BHATTACHARYYA)
        result_2 = cv2.compareHist(hist, temp_hist, cv2.HISTCMP_CHISQR)
        loc = np.where(result >= thresh)
        loc_2 = np.where(result_2 >= thresh_2)

# mark the corresponding location(s)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + temp_w, pt[1] + temp_h), (0, 255, 255), 2)

# mark the corresponding location(s)
for pt in zip(*loc_2[::-1]):
    cv2.rectangle(img_clone, pt, (pt[0] + temp_w, pt[1] + temp_h), (0, 255, 255), 2)


cv2.namedWindow("Results with threshold : 0.6 ", cv2.WINDOW_NORMAL)
cv2.imshow("Results with threshold : 0.6 ", img_rgb)

cv2.namedWindow("Results with threshold : 0.7 ", cv2.WINDOW_NORMAL)
cv2.imshow("Results with threshold : 0.7 ", img_clone)

cv2.waitKey(0)
cv2.destroyAllWindows()