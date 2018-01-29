from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import os
from numpy import genfromtxt
import numpy as np
cwd = os.getcwd()
'''
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
X=np.transpose(X)
'''
cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path
#iris = load_iris()
clf = GradientBoostingClassifier(n_estimators=10, learning_rate=2.0,max_depth=5, random_state=0).fit(X,Y)
scores = cross_val_score(clf, X, Y)
print scores.mean()
