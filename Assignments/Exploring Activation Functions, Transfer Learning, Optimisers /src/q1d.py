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
#             nn.BatchNorm2d(64),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 3 and 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride=1, padding=1),
            activationFunction(),
#             nn.BatchNorm2d(128),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 5,6 and 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride=1, padding=1),
            activationFunction(),
#             nn.BatchNorm2d(256),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 8,9 and 10
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
#             nn.BatchNorm2d(512),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),

            #layer 11,12 and 13
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1, padding=1),
            activationFunction(),
#             nn.BatchNorm2d(512),
            #pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,),  
        )
        
        self.linear=nn.Sequential(
        #Linear layers 14, 15 and 16
#             nn.Linear(512*7*7, 4096),
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
#             img = img.float()
            img=img.to(device)
            out=model(img)
#             target = torch.nn.functional.one_hot(target, num_classes=10)
            target=target.to(device)
            loss=lossFunction(out.float(),target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(i%100==0):
                print(f'\033[30m loss: {loss.item()}')
        print(f' \033[93m epoch {epoch+1} Finished\n\n')
        if(epoch%accuracyEveryNthEpoch==0):
            train_accuracies.append(accuracy(train_loader))
            test_accuracies.append(accuracy(test_loader))
    return(train_accuracies, test_accuracies)
            

batchSize=16

#custom transform pipeline
train_transform=transforms.Compose([
#     transforms.Resize(224),
    transforms.Resize(56),
    transforms.ToTensor(),])

#Getting GPU support if available
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download the MNIST dataset and create a train and test dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=train_transform, download=True)


# only for Q1d C: {0,2,3,5,6,8,9} set to 0 and the set of digits with straight stroke(s), S: {1,4,7}. set to 1
custom_labels = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]   # example custom labels

# Change the target labels of the dataset to your custom labels
train_dataset.targets = torch.tensor([custom_labels[label] for label in train_dataset.targets])
test_dataset.targets = torch.tensor([custom_labels[label] for label in test_dataset.targets])

# Creating a data loader for the train and test datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True)
                     
                     
                     
                     
#1(d) (Transfer Learning) So far, you would have understood the best/optimal
# hyperparameters that yield the best performance in your set of experiments. Save the
# weights/configuration of this model (this is what we will call the pretrained model). In
# this experiment, rather than classifying the images into classes 0 to 9, we will slightly
# change the problem statement in the following way:
# - The dataset with digits 0-9 has mainly two types of information: either the
# handwritten digit is made using a curved stroke or a straight stroke (or maybe both).
# Hence, divide the dataset into two classes viz., the set of digits with curved stroke(s)
# C: {0,2,3,5,6,8,9} and the set of digits with straight stroke(s), S: {1,4,7}.

# using cross entropy loss function
lossFunction=nn.CrossEntropyLoss()
model=VGG16(activationFunction=nn.Tanh)

#loading the pretrained model
state_dict = torch.load('/content/drive/MyDrive/Colab Dataset and models/models/model20')
model.load_state_dict(state_dict)

# Freeze the parameters of convolution layers
for param in model.convolutions.parameters():
    param.requires_grad = False
    
# Freeze the parameters of linear layers
for param in model.linear.parameters():
    param.requires_grad = False

# changing the last layer of the model and setting it's parameters learnable
model.linear[4]=nn.Linear(4096, 2)
for params in model.linear[4].parameters():
    param.requires_grad =True

model=model.to(device)
optimizer=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.1)
l=train(ep=50,accuracyEveryNthEpoch=5)
print(l)


# ([99.96166666666667, 99.99, 99.985, 99.99666666666667, 99.99833333333333, 100.0, 99.99666666666667, 99.99833333333333, 100.0, 100.0], [99.56, 99.61, 99.56, 99.59, 99.6, 99.6, 99.61, 99.6, 99.61, 99.59])



epoch=50
plt.figure(figsize=(9,9))
plt.style.use("seaborn")
plt.title("Learning on the pretrained VGG16 model {optimizer: SGD with momentum, lr: 0.1, loss: crossEntropy, activation: Tanh}")
plt.ticklabel_format(axis='x',useOffset=False)
plt.plot(range(1,epoch,5),l[0], label='Training')
plt.plot(range(1,epoch,5),l[1], label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy%')
plt.legend(loc='lower right')
plt.savefig("accuracies_activation_functions.png")
plt.show()
