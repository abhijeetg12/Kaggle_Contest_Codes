# First XGBoost model for Pima Indians dataset
from numpy import loadtxt

#   from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
import numpy as np
#from xgboost import XGBClassifier
#from sklearn.xgboost import XGBRegressor
# load data
import os
#import xgboost as xgb
from xgboost import XGBClassifier
cwd = os.getcwd()
#path1 = os.path.join(cwd, "../Dataset/mean_imp.csv")
#test = genfromtxt(path1 ,delimiter=',')
#dataset=test
# split data into X and y
#print dataset.shape
#dt=np.transpose(dataset)
#Y=dt[2600]
#X=dt[:2600]
print X.shape
#Y=np.transpose(Y)
#X=np.transpose(X)


X=[1,2,3,4,5,6,7,8,9,0]
Y=[1,2,3,4,5,6,7,8,9,0]
'''
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets'''

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
