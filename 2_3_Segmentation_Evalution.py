import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth,  MiniBatchKMeans
from skimage.color import label2rgb
import random
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt


'''
Function which change each pixel of the images, affected by the probability 
'''
def noise(image,prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
                output[i][j] = image[i][j] + prob*random.uniform(0, 1)*image[i][j] - prob*random.uniform(0, 1)*image[i][j]
    return output

#Loading original image in BGR
original_rgb_img = cv2.imread('E:/PyCharm/PRLM_Lab_47138/Images to use/fire_forest.jpg')


#Add noise to image and repeat the same segmentantion,each time we add 5% noise
noisy_image_5pc = noise(original_rgb_img,0.05)
noisy_image_10pc = noise(original_rgb_img,0.1)
noisy_image_15pc = noise(original_rgb_img,0.15)
noisy_image_20pc = noise(original_rgb_img,0.2)

# Shape of original image and the noisy ones
originShape = original_rgb_img.shape
noisy5pcShape = noisy_image_5pc.shape
noisy10pcShape = noisy_image_10pc.shape
noisy15pcShape = noisy_image_15pc.shape
noisy20pcShape = noisy_image_20pc.shape

# Converting the original image as well the noisy ones into arrays of dimension [nb of pixels in originImage, 3]
# based on r g b intensities (or the 3 chanels that I currnelty have)
flatoriginal=np.reshape(original_rgb_img, [-1, 3])
flatnoisy_5pc_img = np.reshape(noisy_image_5pc, [-1, 3])
flatnoisy_10pc_img = np.reshape(noisy_image_10pc, [-1, 3])
flatnoisy_15pc_img = np.reshape(noisy_image_15pc, [-1, 3])
flatnoisy_20pc_img = np.reshape(noisy_image_20pc, [-1, 3])

#here run the meanshift approach
# Estimate bandwidth for meanshift algorithm the same way and for the noisy images
bandwidthor = estimate_bandwidth(flatoriginal, quantile=0.1, n_samples=100)
msor = MeanShift(bandwidth = bandwidthor, bin_seeding=True)

bandwidth5pc = estimate_bandwidth(flatnoisy_5pc_img, quantile=0.1, n_samples=100)
ms5pc = MeanShift(bandwidth = bandwidth5pc, bin_seeding=True)

bandwidth10pc = estimate_bandwidth(flatnoisy_10pc_img, quantile=0.1, n_samples=100)
ms10pc = MeanShift(bandwidth = bandwidth10pc, bin_seeding=True)

bandwidth15pc = estimate_bandwidth(flatnoisy_15pc_img, quantile=0.1, n_samples=100)
ms15pc = MeanShift(bandwidth = bandwidth15pc, bin_seeding=True)

bandwidth20pc = estimate_bandwidth(flatnoisy_20pc_img, quantile=0.1, n_samples=100)
ms20pc = MeanShift(bandwidth = bandwidth20pc, bin_seeding=True)


# Performing meanshift on flatImg as well to noisy ones
print('Using MeanShift algorithm to the images')
msor.fit(flatoriginal)
ms5pc.fit(flatnoisy_5pc_img)
ms10pc.fit(flatnoisy_10pc_img)
ms15pc.fit(flatnoisy_15pc_img)
ms20pc.fit(flatnoisy_20pc_img)

labels = msor.labels_
labels_5pc = ms5pc.labels_
labels_10pc = ms10pc.labels_
labels_15pc = ms15pc.labels_
labels_20pc = ms20pc.labels_

# Finding and diplaying the number of clusters
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
cluster_centers = msor.cluster_centers_

labels_unique_5pc = np.unique(labels_5pc)
n_clusters_5pc = len(labels_unique_5pc)
cluster_centers_5pc = ms5pc.cluster_centers_

labels_unique_10pc = np.unique(labels_10pc)
n_clusters_10pc = len(labels_unique_10pc)
cluster_centers_10pc = ms10pc.cluster_centers_

labels_unique_15pc = np.unique(labels_15pc)
n_clusters_15pc = len(labels_unique_15pc)
cluster_centers_15pc = ms5pc.cluster_centers_

labels_unique_20pc = np.unique(labels_20pc)
n_clusters_20pc = len(labels_unique_20pc)
cluster_centers_20pc = ms5pc.cluster_centers_

print("number of estimated clusters for the original image : %d", n_clusters_)
print("number of estimated clusters for the noisy image 5% : %d", n_clusters_5pc)
print("number of estimated clusters for the noisy image 10% : %d", n_clusters_10pc)
print("number of estimated clusters for the noisy image 15% : %d", n_clusters_15pc)
print("number of estimated clusters for the noisy image 20% : %d", n_clusters_20pc)

# Reshaping the segmented images to rgb arrays int8
segmentedMeanShift_img = np.reshape(labels, originShape[:2])
segmentedMeanShift_img = label2rgb(segmentedMeanShift_img)
segmentedMeanShift_img = np.array(segmentedMeanShift_img, dtype=np.uint8)

segmentedNoisy5pc = np.reshape(labels, noisy5pcShape[:2])
segmentedNoisy5pc = label2rgb(segmentedNoisy5pc)
segmentedNoisy5pc = np.array(segmentedNoisy5pc, dtype=np.uint8)

segmentedNoisy10pc = np.reshape(labels, noisy10pcShape[:2])
segmentedNoisy10pc = label2rgb(segmentedNoisy10pc)
segmentedNoisy10pc = np.array(segmentedNoisy10pc, dtype=np.uint8)

segmentedNoisy15pc = np.reshape(labels, noisy15pcShape[:2])
segmentedNoisy15pc = label2rgb(segmentedNoisy15pc)
segmentedNoisy15pc = np.array(segmentedNoisy15pc, dtype=np.uint8)

segmentedNoisy20pc = np.reshape(labels, noisy20pcShape[:2])
segmentedNoisy20pc = label2rgb(segmentedNoisy20pc)
segmentedNoisy20pc = np.array(segmentedNoisy20pc, dtype=np.uint8)

#now go for the kmeans
print('Using kmeans algorithm:')
km = MiniBatchKMeans(n_clusters = n_clusters_)
km.fit(flatoriginal)
labels = km.labels_

km5pc = MiniBatchKMeans(n_clusters = n_clusters_5pc)
km5pc.fit(flatnoisy_5pc_img)
labels_5pc = km5pc.labels_

km10pc = MiniBatchKMeans(n_clusters = n_clusters_10pc)
km10pc.fit(flatnoisy_10pc_img)
labels_10pc = km10pc.labels_

km15pc = MiniBatchKMeans(n_clusters = n_clusters_15pc)
km15pc.fit(flatnoisy_15pc_img)
labels_15pc = km15pc.labels_

km20pc = MiniBatchKMeans(n_clusters = n_clusters_20pc)
km20pc.fit(flatnoisy_20pc_img)
labels_20pc = km20pc.labels_


# Finding and diplaying the number of clusters
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

labels_unique_5pc = np.unique(labels_5pc)
n_clusters_5pc = len(labels_unique_5pc)

labels_unique_10pc = np.unique(labels_10pc)
n_clusters_10pc = len(labels_unique_10pc)

labels_unique_15pc = np.unique(labels_15pc)
n_clusters_15pc = len(labels_unique_15pc)

labels_unique_20pc = np.unique(labels_20pc)
n_clusters_20pc = len(labels_unique_20pc)

print("number of estimated clusters : ", n_clusters_)
print("number of estimated clusters for the noisy image 5% : ", n_clusters_5pc)
print("number of estimated clusters for the noisy image 10% : ", n_clusters_10pc)
print("number of estimated clusters for the noisy image 15% : ", n_clusters_15pc)
print("number of estimated clusters for the noisy image 20% : ", n_clusters_20pc)


#Calculate the kmeans
km.fit(flatoriginal)
km.fit(flatnoisy_5pc_img)
km.fit(flatnoisy_10pc_img)
km.fit(flatnoisy_15pc_img)
km.fit(flatnoisy_20pc_img)


# Reshaping the segmented images to rgb arrays int8
segmented_Kmean_img = np.reshape(labels, originShape[:2])
segmented_Kmean_img = label2rgb(segmented_Kmean_img) * 255 # need this to work with cv2. imshow
segmented_Kmean_img = np.array(segmented_Kmean_img, dtype=np.uint8)

segmented_Kmean_5pc_img = np.reshape(labels, noisy5pcShape[:2])
segmented_Kmean_5pc_img = label2rgb(segmented_Kmean_5pc_img) * 255 # need this to work with cv2. imshow
segmented_Kmean_5pc_img = np.array(segmented_Kmean_5pc_img, dtype=np.uint8)

segmented_Kmean_10pc_img = np.reshape(labels, noisy5pcShape[:2])
segmented_Kmean_10pc_img = label2rgb(segmented_Kmean_10pc_img) * 255 # need this to work with cv2. imshow
segmented_Kmean_10pc_img = np.array(segmented_Kmean_10pc_img, dtype=np.uint8)

segmented_Kmean_15pc_img = np.reshape(labels, noisy5pcShape[:2])
segmented_Kmean_15pc_img = label2rgb(segmented_Kmean_15pc_img) * 255 # need this to work with cv2. imshow
segmented_Kmean_15pc_img = np.array(segmented_Kmean_15pc_img, dtype=np.uint8)

segmented_Kmean_20pc_img = np.reshape(labels, noisy5pcShape[:2])
segmented_Kmean_20pc_img = label2rgb(segmented_Kmean_20pc_img) * 255 # need this to work with cv2. imshow
segmented_Kmean_20pc_img = np.array(segmented_Kmean_20pc_img, dtype=np.uint8)

#Convert the rgb images to binary in order to metric them with our binary annotaded image
segmentedMeanShift_img = cv2.cvtColor(segmentedMeanShift_img, cv2.COLOR_RGB2GRAY)
seg_tuple, segm_MeanShift_ar = cv2.threshold(segmentedMeanShift_img , 177, 255, cv2.THRESH_BINARY_INV)

segmentedNoisy5pc = cv2.cvtColor(segmentedNoisy5pc, cv2.COLOR_RGB2GRAY)
seg_tuple_5pc, segm_MeanShift_ar_5pc = cv2.threshold(segmentedNoisy5pc , 177, 255, cv2.THRESH_BINARY_INV)

segmentedNoisy10pc = cv2.cvtColor(segmentedNoisy10pc, cv2.COLOR_RGB2GRAY)
seg_tuple_10pc, segm_MeanShift_ar_10pc = cv2.threshold(segmentedNoisy10pc , 177, 255, cv2.THRESH_BINARY_INV)

segmentedNoisy15pc = cv2.cvtColor(segmentedNoisy15pc, cv2.COLOR_RGB2GRAY)
seg_tuple_15pc, segm_MeanShift_ar_15pc = cv2.threshold(segmentedNoisy15pc , 177, 255, cv2.THRESH_BINARY_INV)

segmentedNoisy20pc = cv2.cvtColor(segmentedNoisy20pc, cv2.COLOR_RGB2GRAY)
seg_tuple_20pc, segm_MeanShift_ar_20pc = cv2.threshold(segmentedNoisy20pc , 177, 255, cv2.THRESH_BINARY_INV)

segmented_Kmean_img = cv2.cvtColor(segmented_Kmean_img, cv2.COLOR_RGB2GRAY)
seg_kmeans_tuple, segm_kmean_ar = cv2.threshold(segmented_Kmean_img , 177, 255, cv2.THRESH_BINARY_INV)

segmented_Kmean_5pc_img = cv2.cvtColor(segmented_Kmean_5pc_img, cv2.COLOR_RGB2GRAY)
seg_kmeans_tuple_5pc, segm_kmean_ar_5pc = cv2.threshold(segmented_Kmean_5pc_img , 177, 255, cv2.THRESH_BINARY_INV)

segmented_Kmean_10pc_img = cv2.cvtColor(segmented_Kmean_10pc_img, cv2.COLOR_RGB2GRAY)
seg_kmeans_tuple_10pc, segm_kmean_ar_10pc = cv2.threshold(segmented_Kmean_10pc_img , 177, 255, cv2.THRESH_BINARY_INV)

segmented_Kmean_15pc_img = cv2.cvtColor(segmented_Kmean_15pc_img, cv2.COLOR_RGB2GRAY)
seg_kmeans_tuple_15pc, segm_kmean_ar_15pc = cv2.threshold(segmented_Kmean_15pc_img , 177, 255, cv2.THRESH_BINARY_INV)

segmented_Kmean_20pc_img = cv2.cvtColor(segmented_Kmean_20pc_img, cv2.COLOR_RGB2GRAY)
seg_kmeans_tuple_20pc, segm_kmean_ar_20pc = cv2.threshold(segmented_Kmean_20pc_img , 177, 255, cv2.THRESH_BINARY_INV)


#Here we plot the segmented images compared to each technique
plt.subplot(5,2,1), plt.imshow(segmentedMeanShift_img)
plt.title('Original Segmented Image used Meanshift')
plt.subplot(5,2,2), plt.imshow(segmented_Kmean_img)
plt.title('Original Segmented Image used Kmean')
plt.subplot(5,2,3), plt.imshow(segmentedNoisy5pc)
plt.title('Noisy 5% Segmented Image used Meanshift')
plt.subplot(5,2,4), plt.imshow(segmented_Kmean_5pc_img)
plt.title('Noisy 5% Segmented Image used Kmean')
plt.subplot(5,2,5), plt.imshow(segmentedNoisy10pc)
plt.title('Noisy 10% Segmented Image used Meanshift')
plt.subplot(5,2,6), plt.imshow(segmented_Kmean_10pc_img)
plt.title('Noisy 10% Segmented Image used Kmean')
plt.subplot(5,2,7), plt.imshow(segmentedNoisy15pc)
plt.title('Noisy 15% Segmented Image used Meanshift')
plt.subplot(5,2,8), plt.imshow(segmented_Kmean_15pc_img)
plt.title('Noisy 15% Segmented Image used Kmean')
plt.subplot(5,2,9), plt.imshow(segmentedNoisy20pc)
plt.title('Noisy 20% Segmented Image used Meanshift')
plt.subplot(5,2,10), plt.imshow(segmented_Kmean_20pc_img)
plt.title('Noisy 20% Segmented Image used Kmean')
plt.show()

#Lets insert now our binary image and try to compare it to the segmented ones
binary_annotated = cv2.imread('E:/PyCharm/PRLM_Lab_47138/Images to use/fire_forest_binary_anoted.jpg',0)
bin_tuple, bin_arr = cv2.threshold(binary_annotated , 177, 255, cv2.THRESH_BINARY_INV)

#We convert the 2d Arrays to 1d in order to compare them using f1_score
bin_ar = bin_arr.flatten()
segm_MeanShift_ar = segm_MeanShift_ar.flatten()
segm_kmean_ar = segm_kmean_ar.flatten()
segm_MeanShift_ar_5pc = segm_MeanShift_ar_5pc.flatten()
segm_MeanShift_ar_10pc = segm_MeanShift_ar_10pc.flatten()
segm_MeanShift_ar_15pc = segm_MeanShift_ar_15pc.flatten()
segm_MeanShift_ar_20pc = segm_MeanShift_ar_20pc.flatten()
segm_kmean_ar_5pc = segm_kmean_ar_5pc.flatten()
segm_kmean_ar_10pc = segm_kmean_ar_10pc.flatten()
segm_kmean_ar_15pc = segm_kmean_ar_15pc.flatten()
segm_kmean_ar_20pc = segm_kmean_ar_20pc.flatten()

score1 = f1_score(bin_ar,segm_MeanShift_ar,average='weighted')
score2 = f1_score(bin_ar,segm_MeanShift_ar_5pc,average='weighted')
score3 = f1_score(bin_ar,segm_MeanShift_ar_10pc,average='weighted')
score4 = f1_score(bin_ar,segm_MeanShift_ar_15pc,average='weighted')
score5 = f1_score(bin_ar,segm_MeanShift_ar_20pc,average='weighted')

score6 = f1_score(bin_ar,segm_kmean_ar,average='weighted')
score7 = f1_score(bin_ar,segm_kmean_ar_5pc,average='weighted')
score8 = f1_score(bin_ar,segm_kmean_ar_10pc,average='weighted')
score9 = f1_score(bin_ar,segm_kmean_ar_15pc,average='weighted')
score10 = f1_score(bin_ar,segm_kmean_ar_20pc,average='weighted')

print("The score between the binary annotated and the Meanshift segmented image is : ", score1)
print("The score between the binary annotated and the Meanshift Segmented Noisy 5% Image is : ", score2)
print("The score between the binary annotated and the Meanshift Segmented Noisy 10% Image is : ", score3)
print("The score between the binary annotated and the Meanshift Segmented Noisy 15% Image is : ", score4)
print("The score between the binary annotated and the Meanshift Segmented Noisy 20% Image is : ", score5)
print("The score between the binary annotated and the Kmean segmented image is : ", score6)
print("The score between the binary annotated and the Kmean Segmented Noisy 5% Image is : ", score7)
print("The score between the binary annotated and the Kmean Segmented Noisy 10% Image is : ", score8)
print("The score between the binary annotated and the Kmean Segmented Noisy 15% Image is : ", score9)
print("The score between the binary annotated and the Kmean Segmented Noisy 20% Image is : ", score10)