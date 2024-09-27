import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# Creating a VGG16 class
class VGG16(nn.Module):
    def __init__(self, classes=10, activationFunction=nn.ReLU, epoch=50):
        super(VGG16,self).__init__()
        
        self.convolutions=nn.Sequential(
            #layer 1 and 2
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            activationFunction(),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 3 and 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 5,6 and 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 8,9 and 10
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 11,12 and 13
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),  
        )
        
        self.linear=nn.Sequential(
        #Linear layers 14, 15 and 16
            nn.Linear(512*1*1, 4096),
            activationFunction(),
            nn.Linear(4096, 4096),
            activationFunction(),
            nn.Linear(4096, classes),
        )

    def forward(self, input):
        conv_out=self.convolutions(input)
        #flattening
        conv_out=conv_out.reshape(conv_out.shape[0],-1)
        return self.linear(conv_out)
    


def accuracy(loader):
    correct=0
    incorrect=0
    for i, (img,target) in enumerate(loader):
        img=img.to(device)
        out=model(img)
        predicted_digit=torch.argmax(out, dim=1)
        for j,k in zip(predicted_digit, target):
            if(j==k):
                correct+=1
            else:
                incorrect+=1
    print(f'correct: {correct}')
    print(f'incorrect: {incorrect}')
    return (100*correct/(len(loader)*batchSize))
        

def train(ep=50,accuracyEveryNthEpoch=10):
    # Training loop
    test_accuracies=[]
    train_accuracies=[]
    for epoch in range(ep):
        for i, (img,target) in enumerate(train_loader):
            img=img.to(device)
            out=model(img)
            target=target.to(device)
            loss=lossFunction(out.float(),target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(i%1000==0):
                print(f'\033[30m loss: {loss.item()}')
        print(f' \033[93m epoch {epoch+1} Finished\n\n')
        if(epoch%accuracyEveryNthEpoch==0):
            train_accuracies.append(accuracy(train_loader))
            test_accuracies.append(accuracy(test_loader))
    return(train_accuracies, test_accuracies)
            

batchSize=16

#custom transform pipeline
train_transform=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),])

#Getting GPU support if available
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download the MNIST dataset and create a train and test dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=train_transform, download=True)

# Creating a data loader for the train and test datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True)
       
       

#(b)
# (What learning rate is better?) Fix the optimizer, batch size, and activation function,the
# number of epochs (to 50 epochs), train this network first using different learning rates (of
# your choice). Tabulate/Plot and discuss the convergence of the network upon varying
# learning rate.


# using cross entropy loss function
lossFunction=nn.CrossEntropyLoss()
list_of_learningRate=[0.001, 0.0001, 0.00001, 0.000001]
list=[]

for i in list_of_learningRate:
    model=VGG16(activationFunction=nn.ReLU)
    model=model.to(device)
    # using adam optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=i) 
    list.append(train(ep=50,accuracyEveryNthEpoch=5))
    print(list)


epoch=50
list_of_learningRate=[0.001, 0.0001,0.00001, 0.000001]
plt.figure(figsize=(15,11)) 
labels=["lr=0.001", "lr=0.0001", "lr=0.00001", "lr=0.000001"]
#training accuracy plot of different activation functions
for i in range(len(list_of_learningRate)):
    axis=plt.subplot(2,2,i+1)
    plt.title(labels[i])
    plt.ticklabel_format(axis='x',useOffset=False)
    plt.plot(range(1,epoch,5),list[i][0], label='Training')
    plt.plot(range(1,epoch,5),list[i][1], label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy%')
    axis.legend(loc='lower right')
plt.savefig("accuracies_activation_functions.png")
