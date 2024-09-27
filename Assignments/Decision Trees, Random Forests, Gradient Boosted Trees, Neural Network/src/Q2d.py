import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys

class nn:
    def __init__(self, batchSize=100, feature_layer=None, hidden_layer=list(), output_layer=None, eta=0.1, activationFunctions=dict()):

        l=[self.sigmoid]*(len(hidden_layer)+1)
        dl=[self.sigmoidDerivative]*(len(hidden_layer)+1)
        for i in activationFunctions.keys():
            if(str.lower(activationFunctions[i])=='relu'):
                l[i-1]=self.ReLU
                dl[i-1]=self.ReLUDerivative
            elif(str.lower(activationFunctions[i])=='sigmoid'):
                pass

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
        # print(l)
        # print('----------------------------------')
        for i in range(len(x)):
            if x[i]==0.0:
                x[i]=random.random()
                # x[i]=0.0
            elif x[i]>0.0:
                x[i]=1.0
            else:
                x[i]=0.0
        # print(x)
        return x

    def RMS(self,x,y):
        return np.average((y-x)**2)
    
    def RMSderivative(self,x,y):
        return (y-x)
    
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
                    error=self.RMSderivative(o,targets[index])
                    self.backPropogation(error)
                    sumError += self.RMS(o,targets[index])
                self.gradientDescent()    
            print('epoch: {}  RMS: {}'.format(epoch,sumError))
            print('next epoch learning rate: {}'.format(self.eta))
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
            self.weights_[i]+=(self.deltaW_[i]/self.batchSize_)*self.eta
            # self.weights_[i]+=self.deltaW_[i]*self.eta


file=open(sys.argv[3]+'/d.txt', 'w')
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
    global file
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
# setting number of hidden nodes for each layer
hidden_nodes=[[100,100],[100,100]]
# setting the activation functions for each layer
activationFunctions=[dict(),{1:'relu', 2:'relu'}]
accuracy=np.zeros((2,len(hidden_nodes)))
# row one will be the training accuracies
# column two will be test accuracies

time_taken=np.zeros(len(hidden_nodes))
# will store the training time


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
    mod=nn(feature_layer=784,hidden_layer=hidden_nodes[hidden],
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

# # Graph for accuracy vs number of hidden nodes
# ax1=fig.add_subplot(2,1,1)
# ax1.plot(hidden_nodes, accuracy[0], marker="o")
# ax1.plot(hidden_nodes, accuracy[1], marker="o")
# ax1.set_xlabel("#Hidden_nodes")
# ax1.set_ylabel("Accuracy")
# ax1.set_title("Accuracy for training and test data")
# ax1.legend(["training", "test"])

# # Graph for accuracy vs number of hidden nodes
# ax2=fig.add_subplot(2,1,2)
# ax2.plot(hidden_nodes, time_taken, marker="o")
# ax2.set_xlabel("#Hidden_nodes")
# ax2.set_ylabel("Time(sec)")
# ax2.set_title("Time taken by training and test data")
# ax2.legend(["training time"])

# plt.savefig(sys.argv[3]+'/d_Graphs')
# plt.tight_layout()
# print('All the asked plots are saved as Graphs.png')
# # plt.show()

file.close()







# Validation accuracy: 77%
# Time taken to train: 2944.3284707069397
# model sigmoid [100, 100] hidden layers
# ----------------confusion matrix----------------
# [[4930   14  184  234   15    3 1581    0   34    1]
#  [  62 5633   16   75   46    2   35    0    6    3]
#  [  83  106 4128   57 1047    0 1265    0   46    0]
#  [ 626  218   63 5318  468    1  358    0   96    1]
#  [  36   14 1215  142 4097    0 1474    0   21    0]
#  [  47    3   60   18   29 5034   97  478   69  135]
#  [  63    8  231  131  240    0  947    0   23    0]
#  [   1    0    0    0    0  657    3 5050   46  309]
#  [ 150    3  103   24   57   45  239   11 5638    2]
#  [   2    1    0    1    1  258    1  461   21 5548]]
# ------------------------------------------------
# Accuracy: 77.20628677144619%
# ----------------confusion matrix----------------
# [[811   4  36  39   1   0 263   0   5   0]
#  [  7 931   5  12   8   1   3   0   1   0]
#  [ 12  14 678   7 197   0 223   0   9   0]
#  [111  42  12 870  60   1  73   0  17   0]
#  [ 12   7 209  30 682   0 240   0   2   0]
#  [ 10   0   6   4   5 836  17  64  16  28]
#  [ 14   0  34  34  37   0 138   0   1   0]
#  [  0   0   0   0   0 103   1 852   8  48]
#  [ 23   2  20   4  10   8  42   1 938   1]
#  [  0   0   0   0   0  51   0  83   3 922]]
# ------------------------------------------------
# Accuracy: 76.58765876587658%


# Validation accuracy: 87%
# Time taken to train: 573.2644910812378
# ReLU model [100, 100] hidden layers
# ----------------confusion matrix----------------
# [[5067   22   69  207   13    8  844    0   16    1]
#  [   9 5805    7   80   14    2   14    0    4    3]
#  [  84   27 4673   56  467    2  543    0   29    0]
#  [ 224  112   52 5255  160    5  165    0   26    0]
#  [  21   16  671  238 4802    1  433    0   24    0]
#  [   9    1    9    0    4 5586    6  162   32  105]
#  [ 516   14  479  143  507    1 3894    0   92    0]
#  [   0    0    2    0    0  256    5 5557   30  205]
#  [  69    3   38   20   32   23   94   15 5738    5]
#  [   1    0    0    1    1  116    2  266    9 5680]]
# ------------------------------------------------
# Accuracy: 86.76311271854532%
# ----------------confusion matrix----------------
# [[815   4  15  29   0   0 139   0   0   0]
#  [  4 955   3  19   1   1   5   0   1   0]
#  [ 10   5 762  13 103   0 103   0   7   0]
#  [ 48  27  10 861  34   1  42   0   5   0]
#  [  4   6 125  37 763   0  79   0   6   0]
#  [  6   0   1   1   0 912   0  36   7  21]
#  [101   2  78  35  92   0 610   0  17   0]
#  [  0   0   0   0   0  55   1 922   4  44]
#  [ 12   1   6   4   7   5  21   1 952   1]
#  [  0   0   0   1   0  26   0  41   1 933]]
# ------------------------------------------------
# Accuracy: 84.85848584858486%