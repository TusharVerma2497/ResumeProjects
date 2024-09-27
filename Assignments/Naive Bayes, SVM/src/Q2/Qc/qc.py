import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from cvxopt import solvers, matrix
from sklearn import svm
from sklearn import metrics
import time
import sys
import os
# location of training and test data
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]


# Reading training data from pickle
with open(os.path.join(trainDataDir,'train_data.pickle'), 'rb') as file:
    training=pickle.load(file)

# my entry number is 2022aiy7514
# last digit is 4, so the classes I have to train my model is 4 and 0
# taking subset of these 2 classes only
# Also changing the class labels from (0,4) to (-1,+1) 

data={'data':[],'labels':[]}
for i in range(len(training["data"])):
    if(training["labels"][i]==0 or training["labels"][i]==4):
        # print(training["labels"][i])
        data['data'].append(np.asarray(list(map(lambda x:x/255.0,training["data"][i].reshape(3072,order='C'))),dtype='float64'))
        if(training["labels"][i]==0):
            data['labels'].append(-1)
        else:
            data['labels'].append(1)

# linear kernel
st=time.time()
model = svm.SVC(kernel='linear')
model.fit(np.array(data['data']),np.array(data['labels']))
print("\nTime taken in seconds linear model: {}".format(time.time()-st))
print("Number of Support Vectors: {}".format(len(model.support_vectors_)))
# print("\nCoefficients:")
print("w: {}".format(model.coef_[0]))
print("b: {}".format(model.intercept_[0]))

with open('wbSklearn','wb') as f:
    pickle.dump((model.intercept_[0],model.coef_[0]),f)

with open('supportVectorsSklearnlinear','wb') as f:
    pickle.dump(model.support_,f)

# gaussian kernel
st=time.time()
modelGaussian=svm.SVC(kernel='rbf',C=1.0,gamma=0.001)
modelGaussian.fit(np.array(data['data']),np.array(data['labels']))
print("\nTime taken in seconds gaussian model: {}".format(time.time()-st))
print("Number of Support Vectors: {}".format(len(modelGaussian.support_vectors_)))


with open('supportVectorsSklearngaussian','wb') as f:
    pickle.dump(modelGaussian.support_,f)
# correct=0
# # for i in range(len(data['data'])):
# t1=model.predict(data['data'])
# t=modelGaussian.predict(data['data'])
# for i in range(len(t)):
#     if t[i]==data['labels'][i]:
#         correct+=1

# print(correct)
# print(len(t)-correct)

with open(os.path.join(testDataDir,'test_data.pickle'), 'rb') as file:
    training=pickle.load(file)

# my entry number is 2022aiy7514
# last digit is 4, so the classes I have to train my model is 4 and 0
# taking subset of these 2 classes only
# Also changing the class labels from (0,4) to (-1,+1) 

data={'data':[],'labels':[]}
for i in range(len(training["data"])):
    if(training["labels"][i]==0 or training["labels"][i]==4):
        # print(training["labels"][i])
        data['data'].append(np.asarray(list(map(lambda x:x/255.0,training["data"][i].reshape(3072,order='C'))),dtype='float64'))
        if(training["labels"][i]==0):
            data['labels'].append(-1)
        else:
            data['labels'].append(1)


# linear testing
correct=0
t=model.predict(data['data'])
for i in range(len(t)):
    if t[i]==data['labels'][i]:
        correct+=1

print("\nLinear correct predictions: "+str(correct))
print("Linear wrong predictions: "+str(len(t)-correct))
print("Test set accuracy linear Model: {}\n".format(correct*100/len(t)))



#Gaussian testing
correct=0
t=modelGaussian.predict(data['data'])
for i in range(len(t)):
    if t[i]==data['labels'][i]:
        correct+=1
print("\nGaussian correct predictions: "+str(correct))
print("Gaussian wrong predictions: "+str(len(t)-correct))
print("Test set accuracy gaussian Model: {}".format(correct*100/len(t)))



# Time taken in seconds linear model: 21.284178495407104
# Number of Support Vectors: 1494

# Coefficients:
# [-0.4035542  -0.09729948 -0.99785795 ... -0.48395384  0.04069652
#  -0.53002951]
# 1.602738207698284

# Time taken in seconds gaussian model: 12.902670860290527
# Number of Support Vectors: 1743

# Linear correct predictions: 1582
# Linear wrong predictions: 418
# Test set accuracy linear Model: 79.1


# Gaussian correct predictions: 1716
# Gaussian wrong predictions: 284

# Test set accuracy gaussian Model: 85.8

