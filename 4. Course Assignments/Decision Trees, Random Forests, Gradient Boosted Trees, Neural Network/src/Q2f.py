import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
# This is working one somehow

class nn:
    def __init__(self, batchSize=100, feature_layer=None, hidden_layer=list(), output_layer=None, eta=0.1, activationFunctions=dict(), costFunction='rms'):

        l=[self.sigmoid]*(len(hidden_layer)+1)
        dl=[self.sigmoidDerivative]*(len(hidden_layer)+1)
        for i in activationFunctions.keys():
            if(str.lower(activationFunctions[i])=='relu'):
                l[i-1]=self.ReLU
                dl[i-1]=self.ReLUDerivative
            elif(str.lower(activationFunctions[i])=='sigmoid'):
                pass
            else:
                raise Exception('only "relu" and "sigmoid" are valid options for activationFunction parameters')
        
        if (str.lower(costFunction)=='rms'):
            self.costFunction_=self.RMS
            self.costFunctionDerivative_=self.RMSderivative
        elif(str.lower(costFunction)=='bce'):
            self.costFunction_=self.BCE
            self.costFunctionDerivative_=self.BCEderivative
        else:
            raise Exception('only "rms" and "bce" are valid options for costFunction parameters')

        self.activationFunction_=l
        self.activationDerivative_=dl
        self.batchSize_=batchSize
        self.layers=[feature_layer]+hidden_layer+[output_layer]
        self.eta=eta
        self.init_eta_=self.eta
        self.weights_=list()
        self.deltaW_=list()
        self.activations_=list()
        self.neuronInputs_=list()
        self.validationAccuracy_=-1
        self.validation_set_=None

        # initialize weights and biases
        l=self.layers
        for i in range((len(l)-1)):
            # self.weights_.append(np.random.rand(l[i]+1, l[i+1]))
            self.weights_.append(np.random.normal(size=(l[i]+1, l[i+1]),scale=1/np.sqrt(l[i])))
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
            self.activations_.append(np.zeros(l[i]))
            self.neuronInputs_.append(np.zeros(l[i]))
            #initializing bias to zeros
        for i in self.weights_:
            i[i.shape[0]-1]=np.zeros(i.shape[1])

    def resetDeltaW(self):
        self.deltaW_=list()
        l=self.layers
        for i in range((len(l)-1)):
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
    
    def sigmoid(self, l):
        x=l
        # return 1.0 / (1.0 + np.exp(-x))
        for i in range(len(x)):
            if x[i]<0:
                x[i]=np.exp(x[i]) / (1.0 + np.exp(x[i]))
            else:
                x[i]=1.0 / (1.0 + np.exp(-x[i]))
        return x

    def sigmoidDerivative(self,x):
        return x*(1.0 - x)

    def ReLU(self,x):
        return x*(x>0)
    
    def ReLUDerivative(self,l):
        x=l
        for i in range(len(x)):
            if x[i]==0.0:
                x[i]=random.random()
                # x[i]=0.0
            elif x[i]>0.0:
                x[i]=1.0
            else:
                x[i]=0.0
        return x

    def RMS(self,x,y):
        return np.average((y-x)**2)
    
    def RMSderivative(self,x,y):
        return (y-x)
    
    # binary cross entropy
    def BCE(self,l,y):
        x=np.round(l,5)
        for i in range(len(x)):
            if y[i] == 1:
                x[i]= -np.log(x[i])
            else:
                x[i]= -np.log(1-x[i])
        return np.sum(x)/len(x)
    
    def BCEderivative(self,l,y):
        x=np.round(l,5)
        for i in range(len(x)):
            if y[i] == 1:
                x[i]= -np.log(1/x[i])
            else:
                x[i]= np.log(1/(1-x[i]))
        return x

    def feedForward(self,inputs):
        layer_output=list()
        layer_input=list()
        layer_output.append(inputs)
        layer_input.append(inputs)
        #activations from the first layer is the input
        activations=inputs
        for i in zip(self.weights_, self.activationFunction_):
            # calculating input to the neurons
            neuron_input=np.dot(np.append(activations,1),i[0])
            # neuron_input=np.matmul(np.append(activations,1).T,i[0])
            # passing it through the activation function for the given layer
            neuron_output=i[1](neuron_input)
            # activations for next layer is inputs from previous layer
            activations=neuron_output
            # saving the activations from each layers
            layer_output.append(neuron_output)
            layer_input.append(neuron_input)
        self.activations_=layer_output 
        self.neuronInputs_=layer_input
        # print(activations)
        # time.sleep(1)
        return activations
    
    #adaptive learning
    def updateLearningRate(self,epoch):
        self.eta=self.init_eta_/np.sqrt(epoch)
    
    def converge(self):

        mask=np.array([0,1,2,3,4,5,6,7,8,9])
        correct=0
        for v in self.validation_set_:
            modelOutput=self.feedForward(v[0])
            # class with the maximum activation is our predicted class
            modelOutput=np.array([1 if i==max(modelOutput) else 0 for i in modelOutput])
            # creating a class mask for convinience in confusion matrix
            modelOutput=np.dot(mask,modelOutput)
            targetOutput=int(np.dot(mask,v[1]))
            if modelOutput==targetOutput:
                correct+=1
        accuracy=int(correct*100/len(self.validation_set_))
        print('Validation accuracy: {}%'.format(accuracy))
        if(accuracy==self.validationAccuracy_):
            return True
        elif(self.validationAccuracy_<accuracy):
            self.validationAccuracy_=accuracy
        return False

    def train(self,inputs,targets):

        # creating a validation set which will be used in checking convergence
        mask=[random.randint(0,len(inputs)-1) for i in range(4000)]
        self.validation_set_=[(inputs[i],targets[i]) for i in mask]

        epoch=1
        while(self.converge()!=True):
            # epoch
            self.updateLearningRate(epoch)
            for iter in range(int(len(targets)/self.batchSize_)):
                sumError=0
                #batch
                # after each bach reset the deltaW to 0s
                self.resetDeltaW()
                for i in range(self.batchSize_):
                    index=random.randint(0,len(inputs)-1)
                    o=self.feedForward(inputs[index])
                    error=self.costFunctionDerivative_(o,targets[index])
                    self.backPropogation(error)
                    sumError += self.costFunction_(o,targets[index])
                self.gradientDescent()    
            print('epoch: {}  RMS: {}'.format(epoch,sumError))
            print('learning rate used: {}'.format(self.eta))
            epoch+=1
            


    def backPropogation(self,error):
        # moving backwards
        for i in range(len(self.weights_)-1,-1,-1):
            delW=error*self.activationDerivative_[i](self.activations_[i+1])
            # if(type(self.activationFunction_[i])==type(self.sigmoid)):
            #     delW=error*self.activationDerivative_[i](self.activations_[i+1])
            # else:
            #     delW=error*self.activationDerivative_[i](self.neuronInputs_[i+1])
            act=(np.append(self.activations_[i],1)).reshape(self.layers[i]+1,1)
            delW=delW.reshape(1,self.layers[i+1])
            # calculating deltaW for weights and bias of a layer
            self.deltaW_[i]+=np.matmul(act,delW)
            # calculating error for the previous layer
            error=np.dot(delW,self.weights_[i].T)
            # dropping error for bias not, no need to backpropogate it has no back connections
            error=np.delete(error,error.shape[1]-1)


    def gradientDescent(self):
        for i in range(len(self.weights_)):
            self.weights_[i]-=(self.deltaW_[i]/self.batchSize_)*self.eta
            # self.weights_[i]+=self.deltaW_[i]*self.eta


file=open(sys.argv[3]+'/f.txt', 'w')
ohe=OneHotEncoder()
def dataPreperation(data):
    # encoding output as onehotEncoding
    output=ohe.fit_transform(data[['9']])
    output=np.array(output.toarray())
    #removing the output column from the dataframe
    data=data.drop(['9'],axis=1)

    # normalizing the pixels
    mean=np.mean(data, axis=0)
    std=np.std(data,axis=0)
    data=(data-mean)/std
    data=data.to_numpy()
    # data=data/255.0
    return (data,output)


def test(inputs,targets):
    # creating a class mask for convinience in confusion matrix
    mask=np.array([0,1,2,3,4,5,6,7,8,9])
    #initializing the confusion matrix
    confusionMatrix=np.zeros((10,10),dtype=int)
    for i in range(len(targets)):
        modelOutput=mod.feedForward(inputs[i])
        # class with the maximum activation is our predicted class
        modelOutput=np.array([1 if i==max(modelOutput) else 0 for i in modelOutput])
        
        modelOutput=np.dot(mask,modelOutput)
        targetOutput=int(np.dot(mask,targets[i]))
        confusionMatrix[modelOutput][targetOutput]+=1
    # # modelOutput=[round(k) for k in modelOutput]
    print("----------------confusion matrix----------------")
    file.write("----------------confusion matrix----------------\n")
    print(confusionMatrix[:10])
    file.write(str(confusionMatrix[:10]))
    print("------------------------------------------------")
    file.write("\n------------------------------------------------\n\n")
    #calculating the accuracy
    accuracy=np.trace(confusionMatrix)*100/np.sum(confusionMatrix)
    print('Accuracy: {}%'.format(accuracy))
    file.write('Accuracy: {}%\n'.format(accuracy))
    file.flush()
    return accuracy


train_data=None
test_data=None
hidden_nodes=[]
hidden_layers=[]
activationFunctions=[]
# loading the best achitecture from the 2e) solution
try:
    with open('e_bestArchitecture','rb') as f:
        a,b=pickle.load(f)
        hidden_nodes.append(a)
        activationFunctions.append(b)
        hidden_layers=len(hidden_nodes[0])
# if best model file was not found then use the parameters based on the experiments        
except:
    hidden_nodes=[[50,50,50]]
    hidden_layers=[3]
    activationFunctions=[{1:'relu', 2:'relu'}]  


time_taken=np.zeros(len(hidden_nodes))
# will store the training time
accuracy=np.zeros((2,len(hidden_nodes)))

for hidden in range(len(hidden_nodes)):
    
    if(type(train_data)==type(None)):
        # loading training data
        with open(sys.argv[1],'r') as f:
            train_data=pd.read_csv(f)
        # loading test data
        with open(sys.argv[2],'r') as f:
            test_data=pd.read_csv(f)

        train_data,train_output=dataPreperation(train_data)
        test_data,test_output=dataPreperation(test_data)

    # Training the model
    st=time.time()
    mod=nn(feature_layer=784,hidden_layer=hidden_nodes[hidden],costFunction='bce',
        output_layer=10,batchSize=100,activationFunctions=activationFunctions[hidden])
    mod.train(train_data,train_output)
    time_taken[hidden]=time.time()-st

    print("Time taken to train: {}".format(time_taken[hidden]))
    file.write("Time taken to train: {}\n".format(time_taken[hidden]))

    # # saving model
    # with open('model'+str(hidden),'wb') as f:
    #     pickle.dump(mod,f)

    # # loading model
    # with open('model5','rb') as f:
    #     mod=pickle.load(f)


    print('model {} hidden layers'.format(hidden_nodes[hidden]))
    file.write('model {} hidden layers\n'.format(hidden_nodes[hidden]))
    # Training accuracy
    accuracy[0][hidden]=test(train_data,train_output)
    # Test accuracy
    accuracy[1][hidden]=test(test_data,test_output)


# # plotting the graphs
# fig = plt.figure(figsize=(7,12))

# # Graph for accuracy vs number of hidden layers
# ax1=fig.add_subplot(2,1,1)
# ax1.plot(hidden_layers, accuracy[0], marker="o")
# ax1.plot(hidden_layers, accuracy[1], marker="o")
# ax1.set_xlabel("#Hidden_layers of 50 nodes each")
# ax1.set_ylabel("Accuracy")
# ax1.set_title("Accuracy for training and test data")
# ax1.legend(["training", "test"])

# # Graph for accuracy vs number of hidden nodes
# ax2=fig.add_subplot(2,1,2)
# ax2.plot(hidden_layers, time_taken, marker="o")
# ax2.set_xlabel("#Hidden_layers with 50 nodes each")
# ax2.set_ylabel("Time(sec)")
# ax2.set_title("Time taken by training and test data")
# ax2.legend(["training time"])

# plt.savefig(sys.argv[3]+'f_Graphs')
# plt.tight_layout()
# print('All the asked plots are saved as Graphs.png')
# # plt.show()

file.close()


# Validation accuracy: 86%
# Time taken to train: 1273.2066054344177
# model [50, 50, 50] hidden layers
# ----------------confusion matrix----------------
# [[4940   10   68  169    8    1  793    0   15    0]
#  [  16 5783   11   63   10    0   18    0    3    2]
#  [  95   39 4715   43  517    1  656    0   23    0]
#  [ 330  135   68 5378  212    5  225    0   42    0]
#  [  16   14  680  197 4837    0  544    0   20    0]
#  [   8    0    4    0    1 5566    0  139   19   96]
#  [ 541   16  427  133  390    3 3687    0  103    1]
#  [   0    0    1    0    0  275    3 5603   24  189]
#  [  53    3   25   15   24   29   73   17 5747    3]
#  [   1    0    1    2    1  120    1  241    4 5708]]
# ------------------------------------------------
# Accuracy: 86.60811013516891%
# ----------------confusion matrix----------------
# [[802   2  16  24   0   0 145   0   0   0]
#  [  2 953   3  14   2   0   1   0   0   0]
#  [ 14   9 760   5 109   0 121   0   5   0]
#  [ 59  29  17 874  40   1  49   0   6   0]
#  [  6   4 115  36 779   0  94   0  10   0]
#  [  1   0   1   1   0 911   2  28  10  15]
#  [104   3  85  41  64   0 573   0  16   1]
#  [  0   0   0   0   0  52   0 933   4  43]
#  [ 11   0   3   4   6   5  15   1 949   0]
#  [  1   0   0   1   0  31   0  38   0 940]]
# ------------------------------------------------
# Accuracy: 84.74847484748474%