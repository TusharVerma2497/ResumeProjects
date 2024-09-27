import numpy as np
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
import sys


file=open(sys.argv[3]+'/g.txt', 'w')
ohe=OneHotEncoder()
def dataPreperation(data):
    # encoding output as onehotEncoding
    output=ohe.fit_transform(data[['9']])
    output=np.array(output.toarray())
    #removing the output column from the dataframe
    # output=data[['9']].to_numpy()
    data=data.drop(['9'],axis=1)

    # normalizing the pixels
    mean=np.mean(data, axis=0)
    std=np.std(data,axis=0)
    data=(data-mean)/std
    data=data.to_numpy()
    # data=data/255.0
    # return (data,np.ravel(output))
    return (data,output)


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
mod=MLPClassifier(batch_size=100,verbose=True,learning_rate='adaptive',
                    hidden_layer_sizes=(50,50,50),solver='sgd',
                    nesterovs_momentum=False,learning_rate_init=0.1,
                    momentum=0.0)
mod.fit(train_data,train_output) 
st=time.time()-st

print("Time taken to train: {}".format(st))
file.write('model {} hidden layers\n'.format(st))

# # saving model
# with open('model'+str(hidden),'wb') as f:
#     pickle.dump(mod,f)

# # loading model
# with open('model5','rb') as f:
#     mod=pickle.load(f)


print('model {} hidden layers'.format(hidden_nodes[0]))
file.write('model {} hidden layers\n'.format(hidden_nodes[0]))

# Training accuracy
acc=mod.score(train_data,train_output)
print("Training accuracy: {}\n".format(acc))
file.write("Training accuracy: {}\n".format(acc))
# Test accuracy
acc=mod.score(test_data,test_output)
print("Test accuracy: {}\n".format(acc))
file.write("Test accuracy: {}\n".format(acc))


file.close()
# Iteration 200, loss = 0.00206089
# Time taken to train: 906.4067075252533
# model [50, 50, 50, 50] hidden layers
# 0.9999666661111019
# 0.8674878487848785