import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, markers
from mpl_toolkits import mplot3d
import math
import time
import threading



#reading the commandline arguments
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]


# reading data
data=pd.read_csv(trainDataDir+"/X.csv",header=None)
X_1=np.array(data.iloc[:,0])
X_2=np.array(data.iloc[:,1])

# normalizing x1 and x2
X_1=((X_1-X_1.mean())/X_1.std())
X_2=((X_2-X_2.mean())/X_2.std())
Y=pd.read_csv(trainDataDir+"/Y.csv",header=None)
Y=np.array(Y.iloc[:,0])

#initializing thetas to zeros
thetas=np.zeros(3)
X=np.array([X_2,X_1,np.ones(len(X_1))])

def logisticRegression():
    global thetas
    prevLTheta=0.0
    while True:
    # for i in range(1):
        # Z=thetas[0]X_0 + thetas[1]*X_1 + thetas[2]*X_2
        Z=np.matmul(thetas,X) #1*99

        # now performing sigmoid squishification
        hTheta=1.0/(1.0+np.exp(-Z))

        # calculating cost 
        # cost=Y*(-np.log(hTheta))+(Y-1)*(np.log(1-hTheta))
        LTheta=np.multiply(Y,np.log(hTheta)) + np.multiply((1-Y),np.log(1-hTheta))
        LTheta=sum(LTheta)
        # print(cost)

        #Convergence criterai
        if(abs(prevLTheta-LTheta)<0.0000001):
            # print(thetas)
            break
        prevLTheta=LTheta

        # Because of Newton's method we are calculating Hessian Matrix
        # calculating first derivative of the log likelihood function as L'(θ)=X.T(Y-h(θ))
        DeltaThetaJ=np.matmul(X,(Y-hTheta))

        # Calculating Hessian matrix second derivative of the log likelihood function as -X.T(h(θ)(1-h(θ)))
        temp2=(hTheta*(1-hTheta))
        temp=temp2*X
        # Hessian Matrix :
        H=np.matmul(temp,X.T)
        # θk+1 = θk + inverse(H) * L'(θ)
        invH=np.linalg.inv(H)
        delTheta=np.matmul(invH,DeltaThetaJ.T)
        # performing optimization
        thetas=thetas+delTheta

logisticRegression()

# (b) plotting the data and the line of regression
# fig=plt.figure(figsize=(10,9))
# for i in range(len(Y)):
    # if(Y[i]==0):
        # plt.scatter(X[1][i],X[0][i], marker='o', color='red')
    # else:
        # plt.scatter(X[1][i],X[0][i], marker='x', color='blue')
# y=X[0]*thetas[1]+thetas[2]
# plt.plot(X[0],y,color='black')
# plt.show()
# fig.savefig('fig')


#Predicting data
data=pd.read_csv(testDataDir+"/X.csv",header=None)
X_1=np.array(data.iloc[:,0])
X_2=np.array(data.iloc[:,1])

# normalizing x1 and x2
X_1=((X_1-X_1.mean())/X_1.std())
X_2=((X_2-X_2.mean())/X_2.std())

X=np.array([X_2,X_1,np.ones(len(X_1))])
# Z=thetas[0]X_0 + thetas[1]*X_1 + thetas[2]*X_2
Z=np.matmul(thetas,X)
# now performing sigmoid squishification
hTheta=1.0/(1.0+np.exp(-Z))
Y=np.array(list(map(lambda x: 1 if x>0.5 else 0,hTheta)))
pd.DataFrame(Y).to_csv('result_3.txt',index=False,header=None)
