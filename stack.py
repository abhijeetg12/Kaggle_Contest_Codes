from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import svm
import os
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cwd = os.getcwd()
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
X=np.transpose(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

clf1 = BaggingClassifier(KNeighborsClassifier(),max_samples=.5, max_features=.5, n_estimators=100)
clf2 = GradientBoostingClassifier(n_estimators=10, learning_rate=2.0,max_depth=5, random_state=0)
#clf3 = AdaBoostClassifier(algorithm='SAMME', base_estimator=clf1, n_estimators=100, learning_rate=.1  , random_state=None)
clf4 = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=20, max_features=None)
clf5 = svm.SVC(kernel='rbf', gamma=0.0005,C=3)
'''clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf4.fit(X_train,y_train)
clf5.fit(X_train,y_train)'''

'''
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
'''
sclf = StackingClassifier(classifiers=[clf1, clf2, clf4], meta_classifier=clf5)
#accuracy_score(y_true, y_pred, normalize=False)

scores = cross_val_score(sclf, X, Y,cv=2)
print scores
print scores.mean()
