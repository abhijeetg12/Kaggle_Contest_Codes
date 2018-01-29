from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from numpy import genfromtxt
import numpy as np

cwd = os.getcwd()
'''path1 = os.path.join(cwd, "../Dataset/mean_imp.csv")
test = genfromtxt(path1 ,delimiter=',')
dataset=test
# split data into X and y
print dataset.shape
dt=np.transpose(dataset)
Y=dt[2600]
X=dt[:2600]
print X.shape
Y=np.transpose(Y)
X=np.transpose(X)
'''
#scwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path
#iris = load_iris()
clf = BaggingClassifier(KNeighborsClassifier(),max_samples=.5, max_features=.5, n_estimators=100)
scores = cross_val_score(clf, X, Y)
print scores
print scores.mean()

#bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
