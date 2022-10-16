import cv2
import numpy as np
import os



file ="E:/PyCharm/PRLM_Lab_47138/Datasets/Output/validation/"
out_path = "E:/PyCharm/PRLM_Lab_47138/Datasets/UNet_Output/validation"
Images = []

for get_image_from_folder in os.listdir(file):
    image = cv2.imread(os.path.join(file, get_image_from_folder), 0)  # inserting every images in the list
    if image is not None:  # if there is an image append the image in the list
        Images.append(image)
    bin_tuple, TmpMask = cv2.threshold(image, np.mean(image), 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(out_path, get_image_from_folder), TmpMask)
