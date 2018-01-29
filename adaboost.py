from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
import os
from numpy import genfromtxt
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score

clf1 = svm.SVC(kernel='rbf', gamma=0.003,C=2.5)
cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path2 ,delimiter=',')
pca = PCA(n_components=50   )
datatr=pca.fit_transform(X)

#iris = load_iris()
clf = AdaBoostClassifier(algorithm='SAMME', base_estimator=clf1, n_estimators=100, learning_rate=.01    , random_state=None)
clf2=AdaBoostClassifier(algorithm='SAMME', base_estimator=clf1, n_estimators=100, learning_rate=.01    , random_state=None).fit(datatr,Y)
scores = cross_val_score(clf, datatr, Y)
print scores
print scores.mean()
y2=clf2.predict(datatr)
print f1_score(Y, y2, average='macro')
print scores
print scores.mean()
