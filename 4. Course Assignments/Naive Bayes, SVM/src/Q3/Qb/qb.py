import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from cvxopt import solvers, matrix
from sklearn import svm
from sklearn import metrics
import time
import os
import sys


# location of training and test data
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]


# Reading training data from pickle
with open(os.path.join(trainDataDir,'train_data.pickle'), 'rb') as file:
    data=pickle.load(file)

# Flattening the pixels
l=[]
for i in range(len(data['data'])):
    t=list(map(lambda x:x/255.0,data["data"][i].reshape(3072)))
    l.append(t)
data['data']=np.array(l,dtype='float64')

# Training the model
st=time.time()
modelGaussian=svm.SVC(kernel='rbf',C=1.0,gamma=0.001,decision_function_shape='ovo')
modelGaussian.fit(data['data'],np.array(data['labels']))
print("Time taken to train: {}".format(time.time()-st))

# with open('models','wb') as file:
#     pickle.dump(modelGaussian,file)

# with open('models','rb') as file:
#     modelGaussian=pickle.load(file)

# Testing data
with open(os.path.join(testDataDir,'test_data.pickle'), 'rb') as file:
    data=pickle.load(file)

# Flattening the pixels
l=[]
for i in range(len(data['data'])):
    t=list(map(lambda x:x/255.0,data["data"][i].reshape(3072)))
    l.append(t)
data['data']=np.array(l,dtype='float64')

correct=0
t=modelGaussian.predict(data['data'])
for i in range(len(t)):
    if t[i]==data['labels'][i]:
        correct+=1

print("Correct classifications: "+str(correct))
print("Incorrect Classifications: "+str(len(t)-correct))

print('Accuracy of Test Data: {}'.format(correct*100/(len(data['data']))))





# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1592
# obj = -1635.793690, rho = 4.589296
# nSV = 1972, nBSV = 1819
# .
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1806
# obj = -1934.783211, rho = -1.564974
# nSV = 2225, nBSV = 2072
# .
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1564
# obj = -1564.564722, rho = 1.161950
# nSV = 1852, nBSV = 1695
# .
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1424
# obj = -1517.369442, rho = -2.192550
# nSV = 1743, nBSV = 1620
# .
# Warning: using -h 0 may be faster
# **.*
# optimization finished, #iter = 1658
# obj = -1477.317840, rho = -6.084612
# nSV = 1865, nBSV = 1687
# .
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1818
# obj = -1733.157408, rho = -3.747803
# nSV = 2164, nBSV = 1976
# .
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1506
# obj = -1365.696762, rho = -7.833765
# nSV = 1725, nBSV = 1561
# .
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1985
# obj = -2392.907333, rho = 5.071612
# nSV = 2756, nBSV = 2606
# .
# Warning: using -h 0 may be faster
# *.*
# optimization finished, #iter = 2089
# obj = -3041.879201, rho = -1.951165
# nSV = 3351, nBSV = 3249
# .
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 1967
# obj = -2263.826067, rho = -6.491965
# nSV = 2631, nBSV = 2490
# Total nSV = 8450
# [LibSVM]Time taken to train: 141.9613401889801
# Correct classifications: 2965
# Incorrect Classifications: 2035
# Accuracy of Test Data: 59.3