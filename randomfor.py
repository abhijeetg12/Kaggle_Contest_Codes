from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from numpy import genfromtxt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path2 ,delimiter=',')
pca = PCA(n_components=50)
#t=pca.fit(testa)
datatr=pca.fit_transform(X)


#iris = load_iris()
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, max_features=None)
scores = cross_val_score(clf, datatr, Y)
clf2= RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, max_features=None).fit(datatr,Y)
y2=clf2.predict(datatr)
print scores
print scores.mean()

print f1_score(Y, y2, average='macro')
