import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import time

#reading the commandline arguments
testDataDir=sys.argv[1]
st=time.time()
l={'x':[],'y':[],'z':[]}
batch=1

# (a) Sampling 1 million training data using θ = [3,1,2], X1=N(3,4), X2=N(-1,4) and ε = N(0,2) 
X1=np.random.normal(loc=3.0, scale=2.0, size=1000000)
X2=np.random.normal(loc=-1.0, scale=2.0, size=1000000)
epsilon=np.random.normal(loc=0.0, scale=math.sqrt(2), size=1000000)
Y=3+X1+2*X2+epsilon
del epsilon

# (b) implementing SGD algorithm
def SGD(ita=0.001, batchSize=1):
    thetas=np.zeros(3)
    global X1, X2, Y, st
    Jtheta=0
    PreJtheta=0
    preThetas=np.zeros(3)
    j=0
    iterations=0
    st=time.time()
    while(True):
        temp=[]
        tempX1=[]
        tempX2=[]
        for i in range(batchSize):
            # Calculating Hypothesis h(θ)
            hTheta = thetas[0] + thetas[1]*X1[j] + thetas[2]*X2[j]
            temp.append(hTheta-Y[j])
            tempX1.append(X1[j])
            tempX2.append(X2[j])
            j=(j+1)%len(X1) #Round Robin fashion
        iterations+=1
        temp=np.array(temp)

        # updating the thetas
        thetas[0]-=ita*temp.mean()
        thetas[1]-=ita*((temp*tempX1).mean())
        thetas[2]-=ita*((temp*tempX2).mean())
        l['x'].append(thetas[0])
        l['y'].append(thetas[1])
        l['z'].append(thetas[2])

        Jtheta=pow(temp,2).mean()/2
        # print(Jtheta)


        #updating ita
        ita=1.0/(0.5*iterations + 10)
        #if(time.time()-st>20):
            # thetas[0]+=np.random.normal(loc=0.0, scale=0.2)
            # thetas[1]+=np.random.normal(loc=0.0, scale=0.2)
            # thetas[2]+=np.random.normal(loc=0.0, scale=0.2)
            #print('For r = {}\nθ₀: {}\nθ₁: {}\nθ₂: {}\niterations: {}'.format(batchSize,thetas[0],thetas[1],thetas[2],iterations))
            #print(ita)
            #st=time.time()

        # exit condition
        # if(abs(Jtheta-PreJtheta)<0.000000001*iterations):
        # if(abs(Jtheta-PreJtheta)<0.000000001*batchSize):
        if(abs(Jtheta-PreJtheta)<0.0000000001*(batchSize+iterations)):
            #print('For r = {}\nθ₀: {}\nθ₁: {}\nθ₂: {}\niterations: {}'.format(batchSize,thetas[0],thetas[1],thetas[2],iterations))
            return thetas[:]

        PreJtheta=Jtheta
        preThetas[0]=thetas[0]
        preThetas[1]=thetas[1]
        preThetas[2]=thetas[2]

# Calling SGD for different values of r given in the question
model=dict()
def startHere():
    global l, batch
    l1=[1,100]
    #l1=[1000000]
    # l1=list(range(1,100))
    global model
    for batch in l1:
        # time.sleep(4)
        l={'x':[0.0],'y':[0.0],'z':[0.0]}
        st1=time.time()
        model[batch]=SGD(batchSize=batch)
        #print("Time taken: {}\n".format(time.time()-st1))
        
        #plotting graph
        #fig = plt.figure(figsize=(9,9))
        #ax1=fig.add_subplot(1,1,1,projection='3d')
        #ax1.legend(['J(θ): '+str(l['z'][-1]),'Iterations: '+str(len(l['z']))])
        #ax1.plot(l['x'],l['y'],l['z'], 'black', marker='.')
        #plt.show()
        #fig.savefig('fig'+str(batch))

startHere()

# (c) reading independent variables from X.csv and dependent variable from Y.csv and normalizing them
data=pd.read_csv(testDataDir+"/X.csv",header=None)
X_1=np.array(data.iloc[:,0])
X_2=np.array(data.iloc[:,1])
#testing for only batchSize 100 thetas
Predicted_Y= model[1][0] + model[1][1]*X_1 + model[1][2]*X_2
pd.DataFrame(Predicted_Y).to_csv('result_2.txt',index=False,header=None)

