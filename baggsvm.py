from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import os
from numpy import genfromtxt
import numpy as np
from sklearn.metrics import f1_score
'''cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/mean_imp.csv")
test = genfromtxt(path1 ,delimiter=',')
dataset=test
#clf1 = svm.SVC(kernel='rbf', gamma=0.05,C=3)

# split data into X and y
print dataset.shape
dt=np.transpose(dataset)
Y=dt[2600]
X=dt[:2600]
print X.shape
Y=np.transpose(Y)
X=np.transpose(X)

cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/mean_imp.csv")
test = genfromtxt(path1 ,delimiter=',')
dataset=test
# split data into X and y

print dataset.shape
dt=np.transpose(dataset)
Y=dt[2600]
X=dt[:2600]
print X.shape
Y=np.transpose(Y)
X=np.transpose(X)'''

cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path2 ,delimiter=',')

pca = PCA(n_components=40)
#t=pca.fit(testa)
datatr=pca.fit_transform(X)
clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.385, store_covariance=False, store_covariances=None, tol=0.001)
clf2= QuadraticDiscriminantAnalysis(priors=None, reg_param=0.385, store_covariance=False, store_covariances=None, tol=0.001).fit(datatr,Y)
#iris = load_iris()
clf = BaggingClassifier(clf, n_estimators=1000) #,max_samples=1.0, max_features=1.0
scores = cross_val_score(clf, datatr, Y)
clft = BaggingClassifier(clf2, n_estimators=1000).fit(datatr,Y) #max_samples=1.0, max_features=1.0,

y2=clft.predict(datatr)
print f1_score(Y, y2, average='macro')
print scores
print scores.mean()

#bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
