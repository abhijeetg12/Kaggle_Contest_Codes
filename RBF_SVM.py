'''
--------------
RBF KERNEL SVM
--------------
'''

import numpy as np
from sklearn.svm import SVC # using a high level implementation of libsvm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score


X=np.loadtxt('train_features_mean.csv')
y=np.loadtxt('train_labels.csv')
test=np.loadtxt('test_features_mean.csv')

scaler=preprocessing.StandardScaler().fit(X)
X_1=scaler.transform(X)
X1,y1=shuffle(X_1,y,random_state=10)
test1=scaler.transform(test)

Xtrain,Xtest,ytrain,ytest=train_test_split(X1,y1,test_size=0.35,train_size=0.65)

'''
#RBF Kernel
gaus=SVC(gamma=0.0001,kernel='rbf',verbose=False)
#c_list=[0.001,0.01,0.1,1,2,3,4,5,6]
c_list=[6,10,15]
gamma_list=[0.00001,0.0001,0.001,0.01,0.1,1,2,3]
i1=0
score=[]
k=0
for i in c_list : 
	gaus.C=i
	score1=cross_val_score(gaus,Xtrain,ytrain,cv=5)
	score.append(np.mean(score1))
	if k==0 : 
		i1=i
	
	if k>0 and np.mean(score1)>score[k-1]  : 
		i1=i		
	print(k,i,':',np.mean(score1))
	k=k+1

	(0, 0.001, ':', 0.050526316733652821)
(1, 0.01, ':', 0.050526316733652821)
(2, 0.1, ':', 0.17033115561428608)
(3, 1, ':', 0.30105222308953561)
(4, 2, ':', 0.32842142843821265)
(5, 3, ':', 0.34151523879977924)
(6, 4, ':', 0.34669385624048055)
(7, 5, ':', 0.34799426015809448)
(8, 6, ':', 0.34978762262317659)
'''

#Polynomial Kernel

poly=SVC(degree=9,coef0=5,kernel='poly',verbose=False)
#deg_list=[2,3,4,5,6,7,8,9]
#deg_list=[4,5,6,7,8,9]
#coef_list=[0,1,2,3,4,5]
score=[]
i1,j1=[0,0] # To store the degree and coefficient
k=0 # indices for list score
c_list=[0.001,0.01,0.1,1,2,3,4,5,6]

'''for i in deg_list : 
	poly.degree=i
	for j in coef_list : 
		print k
		poly.coef0=j
		score1=cross_val_score(poly,Xtrain,ytrain,cv=5)
		score.append(np.mean(score1))
		if k==0 : 
			i1=i
			j1=j
		if k>0 and np.mean(score1)>score[k-1]  : 
			i1=i
			j1=j
		print(i,j,np.mean(score1))
		k=k+1'''
for i in c_list : 
	poly.C=i
	score1=cross_val_score(poly,Xtrain,ytrain,cv=5)
	score.append(np.mean(score1))
	if k==0 : 
		i1=i
	
	if k>0 and np.mean(score1)>score[k-1]  : 
		i1=i		
	print(k,i,':',np.mean(score1))


'''0
(2, 0, 0.054740548179391016)
1
(2, 1, 0.1493362573941846)
2
(2, 2, 0.18009567595229553)
3
(2, 3, 0.20877029208048059)
4
(2, 4, 0.22577737154167066)
5
(2, 5, 0.23563799082651879)
6
(3, 0, 0.06898908716236804)
7
(3, 1, 0.18738561541398308)
8
(3, 2, 0.26882980831998954)
9
(3, 3, 0.29326117079991104)
10
(3, 4, 0.31300542060388559)
11
(3, 5, 0.32404165719791489)
12
(4, 0, 0.066404079128739621)
13
'''
'''
0
(4, 0, 0.068981400420224964)
1
(4, 1, 0.22208202976568875)
2
(4, 2, 0.31325926798099984)
3
(4, 3, 0.34403011920717874)
4
(4, 4, 0.35278672822492668)
5
(4, 5, 0.34693545520798563)
6
(5, 0, 0.072868225291551264)
7
(5, 1, 0.24491448887459888)
8
(5, 2, 0.34207518402270681)
9
(5, 3, 0.3543871153025635)
10
(5, 4, 0.35243922553513007)
11
(5, 5, 0.34693114280394721)
12
(6, 0, 0.075465428914367555)
13
(6, 1, 0.25739585387297292)
14
(6, 2, 0.35083410548957727)
15
(6, 3, 0.35405198000399302)
16
(6, 4, 0.35325516485573977)
17
(6, 5, 0.35244573973212817)
18
(7, 0, 0.075780821453600614)
19
(7, 1, 0.26596768698788387)
20
(7, 2, 0.34064458666475)
21
(7, 3, 0.35261013748420156)
22
(7, 4, 0.35340895467379463)
23
(7, 5, 0.35357696126918675)
24
(8, 0, 0.076110713362554178)
25
(8, 1, 0.26791530500719912)
26
(8, 2, 0.33610472183822249)
27
(8, 3, 0.35114977878480358)
28
(8, 4, 0.3534121556093619)
29
(8, 5, 0.35261016514517618)
30
(9, 0, 0.074155278934024932)
31
(9, 1, 0.26127524248625295)
32
(9, 2, 0.3317437688760389)
33
(9, 3, 0.34355542329851163)
34
(9, 4, 0.35244869841694182)
35
(9, 5, 0.35454133860292042)
'''