import numpy as np
from fancyimpute import  KNN
from sklearn import preprocessing

#Had previously saved the downloaded datsets as follows
import os
from numpy import genfromtxt

cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/train_features.csv")
#path2 = os.path.join(cwd, "../Dataset/train_labels.csv")
path3 = os.path.join(cwd, "../Dataset/test_features.csv")
#test = genfromtxt(path1 ,delimiter=',')
#dataset=test
dataset=np.loadtxt(path1)
#dataset=test
# split data into X and y
print dataset.shape
dt=np.transpose(dataset)
Y=dt[2600]
X=dt[:2600]
print X.shape
Y=np.transpose(Y)
X=np.transpose(X)

#y=np.loadtxt(path2)
test=np.loadtxt(path3)

#Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.35,train_size=0.65)

#Xt=KNN(k=3).complete(X)
#testt=KNN(k=3).complete(test)
#test1=KNN(k=3).complete(test)

Xt=KNN(k=3).complete(X)
print shape(Xt)
#testt=Soft.complete(test)
'''
scaler=preprocessing.StandardScaler().fit(Xt)
X1=scaler.transform(Xt)
test1=scaler.transform(testt)
#test2=scaler.transform(test1)
numpy.savetxt("KNNtrain_f.csv", X1, delimiter=",")
numpy.savetxt("KNNtrain_l.csv", Y, delimiter=",")
numpy.savetxt("KNNtest_f.csv", test1, delimiter=",")
'''
