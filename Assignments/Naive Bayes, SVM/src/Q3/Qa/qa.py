import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from cvxopt import solvers, matrix
import time
from itertools import combinations
import sys
import os
# location of training and test data
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]

st=time.time()
# This is the right one
def SVM(X,Y,kernel,C):
    # Number of training examples
    m=len(Y)
    K=[]
    for i in range(m):
        for j in range(m):
            K.append(kernel(X[i],X[j]))
    K=np.array(K).reshape(m,m)
    P=matrix(np.outer(Y,Y)*K)
    q=np.ones(m)
    q=matrix(q*-1)
    t1=np.identity(m)
    t2=t1*-1
    G=matrix(np.append(t2,t1).reshape(2*m,m))
    h=matrix(np.append(np.zeros(m),np.ones(m)*C))
    b=matrix(np.array([0.0]))
    A=np.array(Y).reshape(1,m)
    A=matrix(A,tc='d')

    # print('Parameters calculated')
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h,A,b)
    lambdas=np.ravel(sol['x'])

    indices=np.where((lambdas > 1e-5) & (lambdas <=C))[0] #index of support vectors
    lambdas = lambdas[indices] # sv lamdas
    supportV = X[indices] # sv X
    supportVLabel = Y[indices] # sv Y

    #finding b value
    b=0
    for i in range(len(lambdas)):
        t=np.zeros(len(lambdas))
        for j in range(len(lambdas)):
            t[j]=K[indices[i],indices[j]]
        # t.reshape(len(lambdas))
        b+=supportVLabel[i]-np.sum(lambdas*supportVLabel*t)
    if(len(lambdas)==0):
        b=0.0
    else:
        b/=len(lambdas)
    print("Value of Intercept: "+str(b))
    print("Number of Support Vectors: "+str(len(supportV)))
    return {'lambdas':lambdas,'SV':supportV,'SVLabels':supportVLabel,'intercept':b}

# Reading training data from pickle
with open(os.path.join(trainDataDir,'train_data.pickle'), 'rb') as file:
    training=pickle.load(file)

# Getting unique class values
Klass=np.unique(training['labels'])
Klass=list(map(int,Klass))

# 2 Combinations of all the classes of inputs
comb=list(combinations(Klass,2))

# creating dictionary for saving the model parameter value for each combination of classes
models={i:{} for i in comb}

# seperating the data in the classes as index of the data list
data=[{'pixels':[]} for i in range(len(Klass))]
for i in range(len(training["data"])):
    label=int(training['labels'][i])
    data[label]['pixels'].append(np.asarray(list(map(lambda x:x/255.0, training["data"][i].reshape(3072,order='C'))),dtype='float64'))
    

# defining linear kernel
def gaussianKernel(a,b,gamma=0.001):
    t=a-b
    t=np.linalg.norm(t)**2
    return np.exp(-gamma*t)

# Training nc2 models
for i in models.keys():
    # preparing data for training
    print("training for classes: "+str(i))
    trainData=np.array(data[i[0]]['pixels']+data[i[1]]['pixels'])
    trainLabels=np.array([-1.0 for i in range(len(data[i[0]]['pixels']))]+[1.0 for i in range(len(data[i[1]]['pixels']))])
    # training the model and saving the parameters
    models[i]=SVM(X=trainData,Y=trainLabels,kernel=gaussianKernel,C=1.0)

print("Time Taken in seconds: {}".format(time.time()-st))   

# saving the solution to the convex optimization to the file
with open('models','wb') as file:
    pickle.dump(models,file)

# Reading solution from the pickle file
with open('models','rb') as file:
    models=pickle.load(file)

print('Testing..')
def predict(pixels):
    # votes={i:0 for i in sorted(Klass)}
    votes=np.zeros(5)
    for i in comb:
        s=0.0
        for j in range(len(models[i]['SV'])):
            s+=models[i]['lambdas'][j]*models[i]['SVLabels'][j]*gaussianKernel(pixels,models[i]['SV'][j])
        s+=models[i]['intercept']
        if(s<=0):
            votes[i[0]]+=1
        else:
            votes[i[1]]+=1
    m=max(votes)
    votes=np.array(votes)
    l=np.where(m==votes)
    return max(l[0])

# reading testing data
with open(os.path.join(testDataDir,'test_data.pickle'), 'rb') as file:
    testing=pickle.load(file)

correct=0
for i in range(len(testing['data'])):
    t=predict(np.asarray(list(map(lambda x:x/255.0, testing["data"][i].reshape(3072,order='C'))),dtype='float64'))
    if(t==int(testing['labels'][i][0])):
        correct+=1
print("Correct classifications Test data: "+str(correct))
print("Incorrect Classifications : "+str(len(testing['data'])-correct))

print('Accuracy of Test Data: {}'.format(correct*100/(len(testing['data']))))


# Correct classifications Test data: 2950
# Incorrect Classifications : 2050
# Accuracy of Test Data: 59.0
