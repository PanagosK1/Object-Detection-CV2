import keras
from keras.datasets import mnist

#import secondary functions that will be used very frequent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# saving the trained model
model_name = 'finalCNN_60_40_new.h5'

# loading a trained model & use it over test data
loaded_model = keras.models.load_model(model_name)

# the data, split between train and test sets
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
X_train.resize(60000,28)
x_test.resize(10000,28)

# train and evaluate the classifiers
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print('Accuracy for 80-20 of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, Y_train)))


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, Y_train)
print('Accuracy for 80-20 of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, Y_train)))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
print('Accuracy for 80-20 of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, Y_train)))


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
print('Accuracy for 80-20 of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, Y_train)))


#classification tree
# predict outcomes for test data and calculate the test scores
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(x_test)
#calculate the scores
dtree_acc_train_80_20 = accuracy_score(Y_train, y_pred_train)
dtree_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
dtree_pre_train_80_20 = precision_score(Y_train, y_pred_train, average='macro')
dtree_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
dtree_rec_train_80_20 = recall_score(Y_train, y_pred_train, average='macro')
dtree_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
dtree_f1_train_80_20 = f1_score(Y_train, y_pred_train, average='macro')
dtree_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

#print the scores
print('')
print(' Printing performance scores:')
print('')

print('Accuracy scores for 80_20 of Decision Tree classifier are:',
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
y_pred_test = knn.predict(x_test)
#calculate the scores
knn_acc_train_80_20 = accuracy_score(Y_train, y_pred_train)
knn_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
knn_pre_train_80_20 = precision_score(Y_train, y_pred_train, average='macro')
knn_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
knn_rec_train_80_20 = recall_score(Y_train, y_pred_train, average='macro')
knn_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
knn_f1_train_80_20 = f1_score(Y_train, y_pred_train, average='macro')
knn_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

#print the scores
print('Accuracy scores for 80_20 of K-NN classifier are:',
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
y_pred_test = gnb.predict(x_test)
# calculate the scores
GNB_acc_train_80_20 = accuracy_score(Y_train, y_pred_train)
GNB_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
GNB_pre_train_80_20 = precision_score(Y_train, y_pred_train, average='macro')
GNB_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
GNB_rec_train_80_20 = recall_score(Y_train, y_pred_train, average='macro')
GNB_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
GNB_f1_train_80_20 = f1_score(Y_train, y_pred_train, average='macro')
GNB_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

# print the scores
print('Accuracy scores for 80_20 of GNB classifier are:',
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
y_pred_test = gnb.predict(x_test)
# calculate the scores
SVM_acc_train_80_20 = accuracy_score(Y_train, y_pred_train)
SVM_acc_test_80_20 = accuracy_score(y_test, y_pred_test)
SVM_pre_train_80_20 = precision_score(Y_train, y_pred_train, average='macro')
SVM_pre_test_80_20 = precision_score(y_test, y_pred_test, average='macro')
SVM_rec_train_80_20 = recall_score(Y_train, y_pred_train, average='macro')
SVM_rec_test_80_20 = recall_score(y_test, y_pred_test, average='macro')
SVM_f1_train_80_20 = f1_score(Y_train, y_pred_train, average='macro')
SVM_f1_test_80_20 = f1_score(y_test, y_pred_test, average='macro')

# print the scores
print('Accuracy scores for 80_20 of SVM classifier are:',
      'train: {:.2f}'.format(SVM_acc_train_80_20), 'and test: {:.2f}.'.format(SVM_acc_test_80_20))
print('Precision scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_pre_train_80_20), 'and test: {:.2f}.'.format(SVM_pre_test_80_20))
print('Recall scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_rec_train_80_20), 'and test: {:.2f}.'.format(SVM_rec_test_80_20))
print('F1 scores of SVM classifier are:',
      'train: {:.2f}'.format(SVM_f1_train_80_20), 'and test: {:.2f}.'.format(SVM_f1_test_80_20))
print('')



#Creating an array to save the comparissons of each SSIM
Results = (['Technique name', 'Train Data ratio', 'Accuracy (tr)', 'Precision (tr)', 'Recal(tr)','F1score (tr)', 'Accuracy (te)','Precision (te)', 'Recal(te)', 'F1 score (te)'],
          ['Decision Tree', '60/40', dtree_acc_train_80_20,dtree_pre_train_80_20,dtree_rec_train_80_20,dtree_f1_train_80_20,dtree_acc_test_80_20,dtree_pre_test_80_20,dtree_rec_test_80_20,dtree_f1_test_80_20],
          ['K-NN', '60/40', knn_acc_train_80_20, knn_pre_train_80_20, knn_rec_train_80_20,knn_f1_train_80_20, knn_acc_test_80_20, knn_pre_test_80_20, knn_rec_test_80_20, knn_f1_test_80_20],
          ['GNB','60/40', GNB_acc_train_80_20,GNB_pre_train_80_20,GNB_rec_train_80_20,GNB_f1_train_80_20,GNB_acc_test_80_20,GNB_pre_test_80_20,GNB_rec_test_80_20,GNB_f1_test_80_20],
          ['SVM', '60/40', SVM_acc_train_80_20, SVM_pre_train_80_20, SVM_rec_train_80_20, SVM_f1_train_80_20,SVM_acc_test_80_20, SVM_pre_test_80_20, SVM_rec_test_80_20, SVM_f1_test_80_20],
           )

row=0
col=0

import xlsxwriter
workbook = xlsxwriter.Workbook('Results_CNN-Classification_60-40%_new.xlsx')
worksheet = workbook.add_worksheet()
for col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 in (Results):
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

    row += 1

workbook.close()