import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys

# This is working one

class nn:
    def __init__(self, batchSize=100, feature_layer=None, hidden_layer=list(), output_layer=None, eta=0.1, activationFunctions=dict()):

        l=[self.sigmoid]*(len(hidden_layer)+1)
        dl=[self.sigmoidDerivative]*(len(hidden_layer)+1)
        for i in activationFunctions.keys():
            if(str.lower(activationFunctions[i])=='softmax'):
                l[i-1]=self.softMax
                dl[i-1]=self.softMaxDerivative
            elif(str.lower(activationFunctions[i])=='sigmoid'):
                pass

        self.activationFunction_=l
        self.activationDerivative_=dl
        self.batchSize_=batchSize
        self.layers=[feature_layer]+hidden_layer+[output_layer]
        self.eta=eta
        self.weights_=list()
        self.deltaW_=list()
        self.activations_=list()
        self.predeltaW_=None
        self.validationAccuracy_=0
        self.validation_set_=None

        # initialize weights and biases
        l=self.layers
        for i in range((len(l)-1)):
            self.weights_.append(np.random.rand(l[i]+1, l[i+1]))
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
            self.activations_.append(np.zeros(l[i]))

    def resetDeltaW(self):
        self.deltaW_=list()
        l=self.layers
        for i in range((len(l)-1)):
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
      
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoidDerivative(self,x):
        return x*(1.0 - x)

    def RMS(self,x,y):
        return np.average((y-x)**2)
    
    def RMSderivative(self,x,y):
        return 2*(x-y)
    
    def feedForward(self,inputs):
        layer_output=list()
        layer_output.append(inputs)
        #activations from the first layer is the input
        activations=inputs
        for i in zip(self.weights_, self.activationFunction_):
            # calculating input to the neurons
            neuron_input=np.dot(np.append(activations,1),i[0])
            # passing it through the activation function for the given layer
            neuron_output=i[1](neuron_input)
            # activations for next layer is inputs from previous layer
            activations=neuron_output
            # saving the activations from each layers
            layer_output.append(neuron_output)
        self.activations_=layer_output 
        return activations
    
    #adaptive learning
    def updateLearningRate(self,epoch):
        self.eta=0.1/np.sqrt(epoch)
    
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
        mask=[random.randint(0,len(inputs)-1) for i in range(5000)]
        self.validation_set_=[(inputs[i],targets[i]) for i in mask]

        epoch=1
        while(self.converge()!=True):
            # epoch
            for iter in range(int(len(targets)/self.batchSize_)):
                sumError=0
                self.predeltaW_=self.deltaW_
                #batch
                # after each bach reset the deltaW
                self.resetDeltaW()
                for i in range(self.batchSize_):
                    index=random.randint(0,len(inputs)-1)
                    o=self.feedForward(inputs[index])
                    error=self.RMSderivative(o,targets[index])
                    self.backPropogation(error)
                    sumError += self.RMS(o,targets[index])
                self.gradientDescent()    
            print('epoch: {}  RMS: {}'.format(epoch,sumError))
            self.updateLearningRate(epoch)
            print('updated learning rate: {}'.format(self.eta))
            epoch+=1
            


    def backPropogation(self,error):
        # moving backwards
        for i in range(len(self.weights_)-1,-1,-1):
            delW=error*self.activationDerivative_[i](self.activations_[i+1])
            act=(np.append(self.activations_[i],1)).reshape(self.layers[i]+1,1)
            delW=delW.reshape(1,self.layers[i+1])
            self.deltaW_[i]+=np.matmul(act,delW)
            error=np.dot(delW,self.weights_[i].T)
            error=np.delete(error,error.shape[1]-1)


    def gradientDescent(self):
        for i in range(len(self.weights_)):
            self.weights_[i]-=self.deltaW_[i]*self.eta


file=open(sys.argv[3]+'/c.txt', 'w')
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
hidden_nodes=[5,10,15,20,25]
# hidden_nodes=[2,3]

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
    mod=nn(feature_layer=784,hidden_layer=[hidden_nodes[hidden]],output_layer=10,batchSize=100)
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


# plotting the graphs
fig = plt.figure(figsize=(7,12))

# Graph for accuracy vs number of hidden nodes
ax1=fig.add_subplot(2,1,1)
ax1.plot(hidden_nodes, accuracy[0], marker="o")
ax1.plot(hidden_nodes, accuracy[1], marker="o")
ax1.set_xlabel("#Hidden_nodes")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy for training and test data")
ax1.legend(["training", "test"])

# Graph for accuracy vs number of hidden nodes
ax2=fig.add_subplot(2,1,2)
ax2.plot(hidden_nodes, time_taken, marker="o")
ax2.set_xlabel("#Hidden_nodes")
ax2.set_ylabel("Time(sec)")
ax2.set_title("Time taken by training and test data")
ax2.legend(["training time"])

plt.savefig(sys.argv[3]+'/c_Graphs')
plt.tight_layout()
print('All the asked plots are saved as Graphs.png')
# plt.show()


file.close()









# Validation accuracy: 10%
# epoch: 1  RMS: 0.09227587920448205
# updated learning rate: 0.1
# Validation accuracy: 63%
# epoch: 2  RMS: 0.008382129413451398
# updated learning rate: 0.07071067811865475
# Validation accuracy: 74%
# epoch: 3  RMS: 0.0007726841160379521
# updated learning rate: 0.05773502691896258
# Validation accuracy: 77%
# epoch: 4  RMS: 0.005790636088757385
# updated learning rate: 0.05
# Validation accuracy: 77%
# Time taken to train: 57.03181791305542
# model 0 hidden layers
# ----------------confusion matrix----------------
# [[5133   19  245  447  111    3 2095    1   52    0]
#  [  20 5605   36   66   33    2   27    0    3    1]
#  [ 109  135 4269  100  690    2 1291    0   32    0]
#  [ 313  187   42 5052  246    9  117    0   33    9]
#  [  95   23 1241   41 4087    0 1589    0   87    0]
#  [   1    1    3    3    1 5270    2  516   63  189]
#  [ 206   27   69  277  791    0  666    0   38    0]
#  [   1    0    1    3    1  457    1 5278   50  526]
#  [ 121    3   94   11   40   95  211   14 5630   11]
#  [   1    0    0    0    0  162    1  191   12 5263]]
# ------------------------------------------------
# Accuracy: 77.08961816030268%
# ----------------confusion matrix----------------
# [[857   5  48  91  13   0 335   0   8   0]
#  [  4 926   7   8   3   1   2   0   2   0]
#  [ 18  16 691  25 136   0 227   0   7   0]
#  [ 45  40  10 823  45   0  29   0   7   1]
#  [ 14   6 214  10 692   0 271   0  14   0]
#  [  2   0   0   1   0 863   1  78  12  33]
#  [ 39   5  13  38 104   0  94   0   5   0]
#  [  0   0   0   1   0  79   0 895   7  91]
#  [ 21   2  17   3   7  16  41   0 937   2]
#  [  0   0   0   0   0  41   0  27   1 872]]
# ------------------------------------------------
# Accuracy: 76.5076507650765%
# Validation accuracy: 10%
# epoch: 1  RMS: 0.0026727285770660196
# updated learning rate: 0.1
# Validation accuracy: 61%
# epoch: 2  RMS: 0.0006298436578221001
# updated learning rate: 0.07071067811865475
# Validation accuracy: 77%
# epoch: 3  RMS: 0.00016511449882524647
# updated learning rate: 0.05773502691896258
# Validation accuracy: 80%
# epoch: 4  RMS: 0.001064162116737331
# updated learning rate: 0.05
# Validation accuracy: 81%
# epoch: 5  RMS: 0.0006128627178316284
# updated learning rate: 0.044721359549995794
# Validation accuracy: 82%
# epoch: 6  RMS: 0.025676775942026275
# updated learning rate: 0.040824829046386304
# Validation accuracy: 83%
# epoch: 7  RMS: 0.00020988172215273456
# updated learning rate: 0.03779644730092272
# Validation accuracy: 82%
# epoch: 8  RMS: 0.032808333988745525
# updated learning rate: 0.035355339059327376
# Validation accuracy: 83%
# Time taken to train: 117.32068991661072
# model 1 hidden layers
# ----------------confusion matrix----------------
# [[4945   12  193  245   25    3 1388    0   73    1]
#  [  17 5671   22   97   20   11   17    0    3    2]
#  [ 181   89 4496   82  611    7  745    0   51    0]
#  [ 456  133   75 5154  160   12  329    2   34    5]
#  [  34   13  780  251 4708    4  646    0   47    1]
#  [  16   10   11   21    5 5583    6  361   58  147]
#  [ 225   62  347  116  422    5 2672    0   67    1]
#  [   1    0    3    7    1  166    3 5238   23  138]
#  [ 121    7   73   25   48   51  193   19 5626   12]
#  [   4    3    0    2    0  158    1  380   18 5692]]
# ------------------------------------------------
# Accuracy: 82.97638293971566%
# ----------------confusion matrix----------------
# [[803   4  45  47   4   0 237   0  14   0]
#  [  2 929   7  16   4   0   2   0   0   0]
#  [ 31  18 719  16 122   0 140   0  12   0]
#  [ 78  34  14 854  28   1  61   1   9   0]
#  [  8   4 138  37 752   0 128   0   4   0]
#  [  3   0   2   1   1 911   0  67  10  26]
#  [ 51   7  58  24  83   1 392   0  12   0]
#  [  0   0   0   0   0  39   0 871   7  28]
#  [ 24   2  17   5   6  12  40   2 932   6]
#  [  0   2   0   0   0  36   0  59   0 939]]
# ------------------------------------------------
# Accuracy: 81.02810281028103%
# Validation accuracy: 4%
# epoch: 1  RMS: 0.009683084402496937
# updated learning rate: 0.1
# Validation accuracy: 58%
# epoch: 2  RMS: 0.009099193818276575
# updated learning rate: 0.07071067811865475
# Validation accuracy: 69%
# epoch: 3  RMS: 0.00384511700548982
# updated learning rate: 0.05773502691896258
# Validation accuracy: 71%
# epoch: 4  RMS: 6.372690821479239e-05
# updated learning rate: 0.05
# Validation accuracy: 78%
# epoch: 5  RMS: 0.005681422952160181
# updated learning rate: 0.044721359549995794
# Validation accuracy: 80%
# epoch: 6  RMS: 0.00645496672576047
# updated learning rate: 0.040824829046386304
# Validation accuracy: 80%
# Time taken to train: 305.9366545677185
# model 2 hidden layers
# ----------------confusion matrix----------------
# [[4766    7   64  209   15    6 1053    0   22    3]
#  [   8 5638   11   86   25    6   15    0    9    1]
#  [ 170   57 4324  102  791    0  699    0   41    0]
#  [ 402  202  104 5281  424    4  300    0   38    4]
#  [ 101   39  626  179 4087    1  858    0  121    1]
#  [  21   15    4   17    4 5335    5  297   55  213]
#  [ 466   28  837   99  594   17 3004    1  118    1]
#  [   6    6    1    7    9  377    6 5160   13  235]
#  [  58    7   29   18   50   48   59   17 5557    6]
#  [   2    1    0    2    1  206    1  525   26 5535]]
# ------------------------------------------------
# Accuracy: 81.14635243920732%
# ----------------confusion matrix----------------
# [[764   2  19  30   1   1 172   0   3   0]
#  [  2 934   4  18   1   0   0   0   3   1]
#  [ 28  10 682  17 156   0 122   0  15   0]
#  [ 81  38  16 862  69   1  65   0   7   0]
#  [ 15  11 103  33 659   1 147   0  25   0]
#  [  2   1   0   5   1 872   1  49  10  37]
#  [ 94   0 172  27  95   2 476   0  20   2]
#  [  1   1   0   3   3  67   0 853   3  40]
#  [ 12   3   4   5  15   8  16   3 912   1]
#  [  1   0   0   0   0  48   1  95   2 918]]
# ------------------------------------------------
# Accuracy: 79.32793279327933%
# Validation accuracy: 10%
# epoch: 1  RMS: 0.001754739777656068
# updated learning rate: 0.1
# Validation accuracy: 64%
# epoch: 2  RMS: 0.09644480455143187
# updated learning rate: 0.07071067811865475
# Validation accuracy: 72%
# epoch: 3  RMS: 0.006906891206061544
# updated learning rate: 0.05773502691896258
# Validation accuracy: 78%
# epoch: 4  RMS: 0.017385949875906737
# updated learning rate: 0.05
# Validation accuracy: 79%
# epoch: 5  RMS: 0.04113236430159047
# updated learning rate: 0.044721359549995794
# Validation accuracy: 80%
# epoch: 6  RMS: 0.02032407244540848
# updated learning rate: 0.040824829046386304
# Validation accuracy: 81%
# epoch: 7  RMS: 0.050903976253132074
# updated learning rate: 0.03779644730092272
# Validation accuracy: 82%
# epoch: 8  RMS: 0.020944764056236267
# updated learning rate: 0.035355339059327376
# Validation accuracy: 82%
# Time taken to train: 384.5056827068329
# model 3 hidden layers
# ----------------confusion matrix----------------
# [[4797   46   83  388  144    0 1096    0   14    0]
#  [  33 5651   15   49    5    0   16    0    3    1]
#  [  60   86 4203   59  580    2  517    0   32    5]
#  [ 228  139   37 4965  147    3  122    0   13    0]
#  [  66   38  782  210 4484    0  786    0   32    0]
#  [  12    4   13   32    9 5477   10  166   61  154]
#  [ 718   33  825  273  596    2 3331    0  143    0]
#  [   0    0    0    1    1  307    1 5535   19  247]
#  [  85    3   42   18   34   35  118   14 5669    1]
#  [   1    0    0    5    0  174    3  285   14 5591]]
# ------------------------------------------------
# Accuracy: 82.83971399523325%
# ----------------confusion matrix----------------
# [[765  10  12  60  21   0 193   0   2   0]
#  [  5 932   3  14   1   0   0   0   1   0]
#  [ 15  10 675  10 116   0  99   0   8   1]
#  [ 38  29   9 796  28   1  30   0   3   0]
#  [ 15  11 144  47 725   0 141   0   7   0]
#  [  3   0   1   4   0 890   2  28  16  29]
#  [141   6 145  65 100   0 511   0  26   0]
#  [  0   0   1   0   0  62   0 923   3  47]
#  [ 18   2  10   2   9   7  24   0 933   1]
#  [  0   0   0   2   0  40   0  49   1 921]]
# ------------------------------------------------
# Accuracy: 80.71807180718072%
# Validation accuracy: 16%
# epoch: 1  RMS: 0.08375687494127969
# updated learning rate: 0.1
# Validation accuracy: 58%
# epoch: 2  RMS: 0.0031231525971942065
# updated learning rate: 0.07071067811865475
# Validation accuracy: 69%
# epoch: 3  RMS: 0.0010051370287754765
# updated learning rate: 0.05773502691896258
# Validation accuracy: 77%
# epoch: 4  RMS: 0.0005374178985666473
# updated learning rate: 0.05
# Validation accuracy: 80%
# epoch: 5  RMS: 0.003628275280930001
# updated learning rate: 0.044721359549995794
# Validation accuracy: 81%
# epoch: 6  RMS: 0.00020004264325709185
# updated learning rate: 0.040824829046386304
# Validation accuracy: 82%
# epoch: 7  RMS: 0.19926644584298023
# updated learning rate: 0.03779644730092272
# Validation accuracy: 82%
# Time taken to train: 353.88060450553894
# model 4 hidden layers
# ----------------confusion matrix----------------
# [[5070   16  191  374  118   11 1421    0   89    2]
#  [  19 5666   25   58   14    2   21    0    3    0]
#  [ 152  105 4844   76  979    6  927    1   95    0]
#  [ 279  161   38 5056  201   14  142    0   35   14]
#  [  36   24  554  214 4247    1  647    0   36    2]
#  [   3    8    8    5    2 5412    2  349   25  137]
#  [ 347   11  283  183  382   13 2695    1   31    0]
#  [   0    0    1    0    3  235    1 5269   26  207]
#  [  92    8   54   32   52   56  143   19 5646   10]
#  [   2    1    2    2    2  250    1  361   14 5627]]
# ------------------------------------------------
# Accuracy: 82.55470924515409%
# ----------------confusion matrix----------------
# [[822   3  43  72  25   0 241   0  10   0]
#  [  1 939   4  12   6   0   4   0   1   0]
#  [ 25  20 791  14 191   0 173   0  20   0]
#  [ 54  30  10 816  33   3  35   0   4   1]
#  [  8   7  97  40 659   0 120   0   8   0]
#  [  2   1   0   2   0 893   2  58   8  24]
#  [ 72   0  47  32  74   2 394   0   5   0]
#  [  0   0   0   1   1  45   0 889   6  41]
#  [ 16   0   7  11  10  15  31   2 937   3]
#  [  0   0   1   0   1  42   0  51   1 930]]
# ------------------------------------------------
# Accuracy: 80.7080708070807%