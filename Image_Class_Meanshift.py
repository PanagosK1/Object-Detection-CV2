import cv2
import os
from sklearn.cluster import MeanShift, estimate_bandwidth  # KMeans
import numpy as np
#import secondary functions that will be used very frequent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#important: this code has been tested using the following opencv and supportive libraries versions.
#pip install opencv-python==3.4.2.16
#pip install opencv-contrib-python==3.4.2.16

#-----------------------------------------------------------------------------------------
#---------------- SUPPORTING FUNCTIONS GO HERE -------------------------------------------
#-----------------------------------------------------------------------------------------

# return a dictionary that holds all images category by category.
def load_images_from_folder(folder, inputImageSize ):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat)
            #print(' .. parsing image', cat)
            if img is not None:
                # grayscale it
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #resize it, if necessary
                img = cv2.resize(img, (inputImageSize[0], inputImageSize[1]))

                category.append(img)
        images[filename] = category
        print(' . Finished parsing images. What is next?')
    return images


# Creates descriptors using an approach of your choise. e.g. ORB, SIFT, SURF, FREAK, MOPS, ετψ
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def detector_features(images):
    print(' . start detecting points and calculating features for a given image set')
    detector_vectors = {}
    descriptor_list = []
    #sift = cv2.xfeatures2d.SIFT_create()
    #detectorToUse = cv2.xfeatures2d.SIFT_create()
    detectorToUse = cv2.ORB_create()
    for nameOfCategory, availableImages in images.items():
        print(' . we are in category : ', nameOfCategory)
        features = []
        tmpImgCount = 1
        for img in availableImages: # reminder: val
            kp, des = detectorToUse.detectAndCompute(img, None)
            print(' .. image {:d} contributed :'.format(tmpImgCount), str(len(kp)), ' points of interest')
            tmpImgCount +=1
            descriptor_list.extend(des)
            features.append(des)
        detector_vectors[nameOfCategory] = features
        print(' . finished detecting points and calculating features for a given image set')
    return [descriptor_list, detector_vectors] # be aware of the []! this is ONE output as a list


# A k-means clustering algorithm who takes 2 parameter which is number
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def MeanshiftVisualWordsCreation(k, descriptor_list):
    print(' . calculating central points for the existing feature values.')
    #kmeansModel = KMeans(n_clusters = k, n_init=10)
    estbandwidth = estimate_bandwidth(descriptor_list, quantile=0.1, n_samples=800)
    MeanshiftModels  = MeanShift(bandwidth = estbandwidth)
    MeanshiftModels.fit(descriptor_list)
    visualWords = MeanshiftModels.cluster_centers_ # a.k.a. centers of reference
    print(' . done calculating central points for the given feature set.')
    return visualWords, MeanshiftModels

#Creation of the histograms. To create our each image by a histogram. We will create a vector of k values for each
# image. For each keypoints in an image, we will find the nearest center, defined using training set
# and increase by one its value
def mapFeatureValsToHistogram (DataFeaturesByClass, visualWords, TrainedKmeansModel):
    #depenting on the approach you may not need to use all inputs
    histogramsMeanshList = []
    targetClassList = []
    numberOfBinsPerHistogram = visualWords.shape[0]

    for categoryIdx, featureValues in DataFeaturesByClass.items():
        for tmpImageFeatures in featureValues: #yes, we check one by one the values in each image for all images
            tmpImageHistogram = np.zeros(numberOfBinsPerHistogram)
            tmpIdx = list(TrainedKmeansModel.predict(tmpImageFeatures))
            clustervalue, visualWordMatchCounts = np.unique(tmpIdx, return_counts=True)
            tmpImageHistogram[clustervalue] = visualWordMatchCounts
            # do not forget to normalize the histogram values
            numberOfDetectedPointsInThisImage = tmpIdx.__len__()
            tmpImageHistogram = tmpImageHistogram/numberOfDetectedPointsInThisImage

            #now update the input and output coresponding lists
            histogramsMeanshList.append(tmpImageHistogram)
            targetClassList.append(categoryIdx)

    return histogramsMeanshList, targetClassList

#here we run the code for 80-20


#define a fixed image size to work with
inputImageSize = [200, 200, 3] #define the FIXED size that CNN will have as input

#define the path to train and test files
TrainImagesFilePath ='E:/PyCharm/PRLM_Lab_47138/Datasets/80-20/train'
TestImagesFilePath = 'E:/PyCharm/PRLM_Lab_47138/Datasets/80-20/test'


#load the train images
trainImages = load_images_from_folder(TrainImagesFilePath, inputImageSize)  # take all images category by category for train set

#calculate points and descriptor values per image
trainDataFeatures = detector_features(trainImages)
# Takes the descriptor list which is unordered one
TrainDescriptorList = trainDataFeatures[0]

#create the central points for the histograms using k means.
#here we use a rule of the thumb to create the expected number of cluster centers
numberOfClasses = trainImages.__len__() #retrieve num of classes from dictionary
possibleNumOfCentersToUse = 10 * numberOfClasses
visualWords, TrainedKmeansModel = MeanshiftVisualWordsCreation(possibleNumOfCentersToUse, TrainDescriptorList)


# Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
trainBoVWFeatureVals = trainDataFeatures[1]

#create the train input train output format
trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords, TrainedKmeansModel)
#X_train = np.asarray(trainHistogramsList)
#X_train = np.concatenate(trainHistogramsList, axis=0)
X_train = np.stack(trainHistogramsList, axis= 0)

# Convert Categorical Data For Scikit-Learn
from sklearn import preprocessing

# Create a label (category) encoder object
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(trainTargetsList)
#convert the categories from strings to names
y_train = labelEncoder.transform(trainTargetsList)


# train and evaluate the classifiers
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy for 80-20 of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy for 80-20 of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy for 80-20 of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy for 80-20 of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))


# ----------------------------------------------------------------------------------------
#now run the same things on the test data.
# DO NOT FORGET: you use the same visual words, created using training set.

#clear some space
del trainImages, trainBoVWFeatureVals, trainDataFeatures, TrainDescriptorList

#load the train images
testImages = load_images_from_folder(TestImagesFilePath, inputImageSize)  # take all images category by category for train set

#calculate points and descriptor values per image
testDataFeatures = detector_features(testImages)

# Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
testBoVWFeatureVals = testDataFeatures[1]

#create the test input / test output format
testHistogramsList, testTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords, TrainedKmeansModel)
X_test = np.array(testHistogramsList)
y_test = labelEncoder.transform(testTargetsList)


#classification tree
# predict outcomes for test data and calculate the test scores
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
#calculate the scores
dtree_acc_train_80_20 = accuracy_score(y_train, y_pred_train)
dtree_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
dtree_pre_train_80_20 = precision_score(y_train, y_pred_train, average='macro')
dtree_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
dtree_rec_train_80_20 = recall_score(y_train, y_pred_train, average='macro')
dtree_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
dtree_f1_train_80_20 = f1_score(y_train, y_pred_train, average='macro')
dtree_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

#print the scores
print('')
print(' Printing performance scores:')
print('')

print('Accuracy scores for 80-20 of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_acc_train_80_20), 'and test: {:.2f}.'.format(dtree_acc_test_80_20))
print('Precision scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_pre_train_80_20), 'and test: {:.2f}.'.format(dtree_pre_test_80_20))
print('Recall scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_rec_train_80_20), 'and test: {:.2f}.'.format(dtree_rec_test_80_20))
print('F1 scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_f1_train_80_20), 'and test: {:.2f}.'.format(dtree_f1_test_80_20))
print('')

# knn predictions
#now check for both train and test data, how well the model learned the patterns
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
#calculate the scores
knn_acc_train_80_20 = accuracy_score(y_train, y_pred_train)
knn_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
knn_pre_train_80_20 = precision_score(y_train, y_pred_train, average='macro')
knn_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
knn_rec_train_80_20 = recall_score(y_train, y_pred_train, average='macro')
knn_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
knn_f1_train_80_20 = f1_score(y_train, y_pred_train, average='macro')
knn_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

#print the scores
print('Accuracy scores for 80-20 of K-NN classifier are:',
      'train: {:.2f}'.format(knn_acc_train_80_20), 'and test: {:.2f}.'.format(knn_acc_test_80_20))
print('Precision scores of Logistic regression classifier are:',
      'train: {:.2f}'.format(knn_pre_train_80_20), 'and test: {:.2f}.'.format(knn_pre_test_80_20))
print('Recall scores of K-NN classifier are:',
      'train: {:.2f}'.format(knn_rec_train_80_20), 'and test: {:.2f}.'.format(knn_rec_test_80_20))
print('F1 scores of K-NN classifier are:',
      'train: {:.2f}'.format(knn_f1_train_80_20), 'and test: {:.2f}.'.format(knn_f1_test_80_20))
print('')


#naive Bayes
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
GNB_acc_train_80_20 = accuracy_score(y_train, y_pred_train)
GNB_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
GNB_pre_train_80_20 = precision_score(y_train, y_pred_train, average='macro')
GNB_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
GNB_rec_train_80_20 = recall_score(y_train, y_pred_train, average='macro')
GNB_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
GNB_f1_train_80_20 = f1_score(y_train, y_pred_train, average='macro')
GNB_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

# print the scores
print('Accuracy scores for 80-20 of GNB classifier are:',
      'train: {:.2f}'.format(GNB_acc_train_80_20), 'and test: {:.2f}.'.format(GNB_acc_test_80_20))
print('Precision scores of GBN classifier are:',
      'train: {:.2f}'.format(GNB_pre_train_80_20), 'and test: {:.2f}.'.format(GNB_pre_test_80_20))
print('Recall scores of GNB classifier are:',
      'train: {:.2f}'.format(GNB_rec_train_80_20), 'and test: {:.2f}.'.format(GNB_rec_test_80_20))
print('F1 scores of GNB classifier are:',
      'train: {:.2f}'.format(GNB_f1_train_80_20), 'and test: {:.2f}.'.format(GNB_f1_test_80_20))
print('')


#support vector machines
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
SVM_acc_train_80_20 = accuracy_score(y_train, y_pred_train)
SVM_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
SVM_pre_train_80_20 = precision_score(y_train, y_pred_train, average='macro')
SVM_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
SVM_rec_train_80_20 = recall_score(y_train, y_pred_train, average='macro')
SVM_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
SVM_f1_train_80_20 = f1_score(y_train, y_pred_train, average='macro')
SVM_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

# print the scores
print('Accuracy scores for 80-20 of SVM classifier are:',
      'train: {:.2f}'.format(SVM_acc_train_80_20), 'and test: {:.2f}.'.format(SVM_acc_test_80_20))
print('Precision scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_pre_train_80_20), 'and test: {:.2f}.'.format(SVM_pre_test_80_20))
print('Recall scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_rec_train_80_20), 'and test: {:.2f}.'.format(SVM_rec_test_80_20))
print('F1 scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_f1_train_80_20), 'and test: {:.2f}.'.format(SVM_f1_test_80_20))
print('')

#Lets do the same and for the 60% train 40% test
#define a fixed image size to work with
inputImageSize = [200, 200, 3] #define the FIXED size that CNN will have as input

#define the path to train and test files
TrainImagesFilePath ='E:/PyCharm/PRLM_Lab_47138/Datasets/60-40/train'
TestImagesFilePath = 'E:/PyCharm/PRLM_Lab_47138/Datasets/60-40/test'


#load the train images
trainImages = load_images_from_folder(TrainImagesFilePath, inputImageSize)  # take all images category by category for train set

#calculate points and descriptor values per image
trainDataFeatures = detector_features(trainImages)
# Takes the descriptor list which is unordered one
TrainDescriptorList = trainDataFeatures[0]

#create the central points for the histograms using k means.
#here we use a rule of the thumb to create the expected number of cluster centers
numberOfClasses = trainImages.__len__() #retrieve num of classes from dictionary
possibleNumOfCentersToUse = 10 * numberOfClasses
visualWords, TrainedKmeansModel = MeanshiftVisualWordsCreation(possibleNumOfCentersToUse, TrainDescriptorList)


# Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
trainBoVWFeatureVals = trainDataFeatures[1]

#create the train input train output format
trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords, TrainedKmeansModel)
#X_train = np.asarray(trainHistogramsList)
#X_train = np.concatenate(trainHistogramsList, axis=0)
X_train = np.stack(trainHistogramsList, axis= 0)


# Create a label (category) encoder object
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(trainTargetsList)
#convert the categories from strings to names
y_train = labelEncoder.transform(trainTargetsList)


# train and evaluate the classifiers
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy for 60-40% of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))


clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy for 60-40% of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))


gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy for 60-40% of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))


svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy for 60-40% of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))


# ----------------------------------------------------------------------------------------
#now run the same things on the test data.
# DO NOT FORGET: you use the same visual words, created using training set.

#clear some space
del trainImages, trainBoVWFeatureVals, trainDataFeatures, TrainDescriptorList

#load the train images
testImages = load_images_from_folder(TestImagesFilePath, inputImageSize)  # take all images category by category for train set

#calculate points and descriptor values per image
testDataFeatures = detector_features(testImages)

# Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
testBoVWFeatureVals = testDataFeatures[1]

#create the test input / test output format
testHistogramsList, testTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords, TrainedKmeansModel)
X_test = np.array(testHistogramsList)
y_test = labelEncoder.transform(testTargetsList)


#classification tree
# predict outcomes for test data and calculate the test scores
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
#calculate the scores
dtree_acc_train_60_40 = accuracy_score(y_train, y_pred_train)
dtree_acc_test_60_40 = accuracy_score(y_test, y_pred_test)
dtree_pre_train_60_40 = precision_score(y_train, y_pred_train, average='macro')
dtree_pre_test_60_40 = precision_score(y_test, y_pred_test, average='macro')
dtree_rec_train_60_40 = recall_score(y_train, y_pred_train, average='macro')
dtree_rec_test_60_40 = recall_score(y_test, y_pred_test, average='macro')
dtree_f1_train_60_40 = f1_score(y_train, y_pred_train, average='macro')
dtree_f1_test_60_40 = f1_score(y_test, y_pred_test, average='macro')

#print the scores
print('')
print(' Printing performance scores:')
print('')

print('Accuracy scores for 60-40 of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_acc_train_60_40), 'and test: {:.2f}.'.format(dtree_acc_test_60_40))
print('Precision scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_pre_train_60_40), 'and test: {:.2f}.'.format(dtree_pre_test_60_40))
print('Recall scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_rec_train_60_40), 'and test: {:.2f}.'.format(dtree_rec_test_60_40))
print('F1 scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(dtree_f1_train_60_40), 'and test: {:.2f}.'.format(dtree_f1_test_60_40))
print('')

# knn predictions
#now check for both train and test data, how well the model learned the patterns
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
#calculate the scores
knn_acc_train_60_40 = accuracy_score(y_train, y_pred_train)
knn_acc_test_60_40 = accuracy_score(y_test, y_pred_test)
knn_pre_train_60_40 = precision_score(y_train, y_pred_train, average='macro')
knn_pre_test_60_40 = precision_score(y_test, y_pred_test, average='macro')
knn_rec_train_60_40 = recall_score(y_train, y_pred_train, average='macro')
knn_rec_test_60_40 = recall_score(y_test, y_pred_test, average='macro')
knn_f1_train_60_40 = f1_score(y_train, y_pred_train, average='macro')
knn_f1_test_60_40 = f1_score(y_test, y_pred_test, average='macro')

#print the scores
print('Accuracy scores for 60-40 of K-NN classifier are:',
      'train: {:.2f}'.format(knn_acc_train_60_40), 'and test: {:.2f}.'.format(knn_acc_test_60_40))
print('Precision scores of Logistic regression classifier are:',
      'train: {:.2f}'.format(knn_pre_train_60_40), 'and test: {:.2f}.'.format(knn_pre_test_60_40))
print('Recall scores of K-NN classifier are:',
      'train: {:.2f}'.format(knn_rec_train_60_40), 'and test: {:.2f}.'.format(knn_rec_test_60_40))
print('F1 scores of K-NN classifier are:',
      'train: {:.2f}'.format(knn_f1_train_60_40), 'and test: {:.2f}.'.format(knn_f1_test_60_40))
print('')


#naive Bayes
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
GNB_acc_train_60_40 = accuracy_score(y_train, y_pred_train)
GNB_acc_test_60_40 = accuracy_score(y_test, y_pred_test)
GNB_pre_train_60_40 = precision_score(y_train, y_pred_train, average='macro')
GNB_pre_test_60_40 = precision_score(y_test, y_pred_test, average='macro')
GNB_rec_train_60_40 = recall_score(y_train, y_pred_train, average='macro')
GNB_rec_test_60_40 = recall_score(y_test, y_pred_test, average='macro')
GNB_f1_train_60_40 = f1_score(y_train, y_pred_train, average='macro')
GNB_f1_test_60_40 = f1_score(y_test, y_pred_test, average='macro')

# print the scores
print('Accuracy scores for 60-40 of GNB classifier are:',
      'train: {:.2f}'.format(GNB_acc_train_60_40), 'and test: {:.2f}.'.format(GNB_acc_test_60_40))
print('Precision scores of GBN classifier are:',
      'train: {:.2f}'.format(GNB_pre_train_60_40), 'and test: {:.2f}.'.format(GNB_pre_test_60_40))
print('Recall scores of GNB classifier are:',
      'train: {:.2f}'.format(GNB_rec_train_60_40), 'and test: {:.2f}.'.format(GNB_rec_test_60_40))
print('F1 scores of GNB classifier are:',
      'train: {:.2f}'.format(GNB_f1_train_60_40), 'and test: {:.2f}.'.format(GNB_f1_test_60_40))
print('')


#support vector machines
# now check for both train and test data, how well the model learned the patterns
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
# calculate the scores
SVM_acc_train_60_40 = accuracy_score(y_train, y_pred_train)
SVM_acc_test_60_40 = accuracy_score(y_test, y_pred_test)
SVM_pre_train_60_40 = precision_score(y_train, y_pred_train, average='macro')
SVM_pre_test_60_40 = precision_score(y_test, y_pred_test, average='macro')
SVM_rec_train_60_40 = recall_score(y_train, y_pred_train, average='macro')
SVM_rec_test_60_40 = recall_score(y_test, y_pred_test, average='macro')
SVM_f1_train_60_40 = f1_score(y_train, y_pred_train, average='macro')
SVM_f1_test_60_40 = f1_score(y_test, y_pred_test, average='macro')

# print the scores
print('Accuracy scores for 60-40 of SVM classifier are:',
      'train: {:.2f}'.format(SVM_acc_train_60_40), 'and test: {:.2f}.'.format(SVM_acc_test_60_40))
print('Precision scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_pre_train_60_40), 'and test: {:.2f}.'.format(SVM_pre_test_60_40))
print('Recall scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_rec_train_60_40), 'and test: {:.2f}.'.format(SVM_rec_test_60_40))
print('F1 scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_f1_train_60_40), 'and test: {:.2f}.'.format(SVM_f1_test_60_40))
print('')


#Creating an array to save the comparissons of each SSIM
Results = (['FeatureExtraction','Clustering Detection','Train Data ratio',' Classifier Used','Accuracy (tr)','Precision (tr)','Recal(tr)','F1score (tr)',' Accuracy (te)','Precision (te)','Recal(te)','F1 score (te)'],
          ['ORB','Meanshift','80/20','Decision Tree',dtree_acc_train_80_20,dtree_pre_train_80_20,dtree_rec_train_80_20,dtree_f1_train_80_20,dtree_acc_test_80_20,dtree_pre_test_80_20,dtree_rec_test_80_20,dtree_f1_test_80_20],
          ['ORB','Meanshift','80/20','K-NN',knn_acc_train_80_20,knn_pre_train_80_20,knn_rec_train_80_20,knn_f1_train_80_20,knn_acc_test_80_20,knn_pre_test_80_20,knn_rec_test_80_20,knn_f1_test_80_20],
          ['ORB','Meanshift','80/20','GNB',GNB_acc_train_80_20,GNB_pre_train_80_20,GNB_rec_train_80_20,GNB_f1_train_80_20,GNB_acc_test_80_20,GNB_pre_test_80_20,GNB_rec_test_80_20,GNB_f1_test_80_20],
          ['ORB', 'Meanshift', '80/20', 'SVM', SVM_acc_train_80_20, SVM_pre_train_80_20, SVM_rec_train_80_20, SVM_f1_train_80_20, SVM_acc_test_80_20, SVM_pre_test_80_20, SVM_rec_test_80_20, SVM_f1_test_80_20],
          ['ORB', 'Meanshift', '60/40', 'Decision Tree', dtree_acc_train_60_40, dtree_pre_train_60_40,dtree_rec_train_60_40, dtree_f1_train_60_40, dtree_acc_test_60_40, dtree_pre_test_60_40,dtree_rec_test_60_40, dtree_f1_test_60_40],
          ['ORB', 'Meanshift', '60/40', 'K-NN', knn_acc_train_60_40, knn_pre_train_60_40, knn_rec_train_60_40,knn_f1_train_60_40, knn_acc_test_60_40, knn_pre_test_60_40, knn_rec_test_60_40, knn_f1_test_60_40],
          ['ORB', 'Meanshift', '60/40', 'GNB', GNB_acc_train_60_40, GNB_pre_train_60_40, GNB_rec_train_60_40,GNB_f1_train_60_40, GNB_acc_test_60_40, GNB_pre_test_60_40, GNB_rec_test_60_40, GNB_f1_test_60_40],
          ['ORB', 'Meanshift', '60/40', 'SVM', SVM_acc_train_60_40, SVM_pre_train_60_40, SVM_rec_train_60_40,SVM_f1_train_60_40, SVM_acc_test_60_40, SVM_pre_test_60_40, SVM_rec_test_60_40, SVM_f1_test_60_40])


row=0
col=0
import xlsxwriter


workbook = xlsxwriter.Workbook('Results_Classification_Meanshift.xlsx')
worksheet = workbook.add_worksheet()
for col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 in (Results):
    worksheet.write(row, col,     col1)
    worksheet.write(row, col + 1, col2)
    worksheet.write(row, col + 2, col3)
    worksheet.write(row, col + 3, col4)
    worksheet.write(row, col + 4, col5)
    worksheet.write(row, col + 5, col6)
    worksheet.write(row, col + 6, col7)
    worksheet.write(row, col + 7, col8)
    worksheet.write(row, col + 8, col9)
    worksheet.write(row, col + 9, col10)
    worksheet.write(row, col + 10, col11)
    worksheet.write(row, col + 11, col12)
    row += 1

workbook.close()
