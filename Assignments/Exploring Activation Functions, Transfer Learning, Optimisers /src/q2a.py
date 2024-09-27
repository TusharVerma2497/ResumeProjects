import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import transforms
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from re import TEMPLATE
import torch.nn.functional as F

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



# Define a set of transformations to be applied to the dataset
transform = transforms.Compose([
    transforms.ToTensor(), # Convert the PIL Image to a PyTorch Tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Normalize the image pixel values
])

data_path='./data'
batchSize=16
# Load the CIFAR-10 train and test sets with the defined transformations
train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True)





class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out) + out

        temp= self.layer2(out)
        # padding is added to make dimensions compatible
        out = F.pad(out[:,:,::2,::2], (0,0,0,0,out.shape[1]//2, out.shape[1]//2,0,0))
        out=temp+out

        temp= self.layer3(out)
        out = F.pad(out[:,:,::2,::2], (0,0,0,0,out.shape[1]//2, out.shape[1]//2,0,0))
        out=temp+out

        temp= self.layer4(out)
        out = F.pad(out[:,:,::2,::2], (0,0,0,0,out.shape[1]//2, out.shape[1]//2,0,0))
        out=temp+out
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


lossFunction=nn.CrossEntropyLoss()
lr=[0.01,0.001,0.0001,0.00001]
list=[]
for i in lr:
    #instantiating resnet18 from scratch
    model=ResNet18()
    #loading the pretrained weights of imagenet dataset in out implementation of the resnet18
    resnet18= models.resnet18(pretrained=True)
    model.conv1 = resnet18.conv1
    model.bn1 = resnet18.bn1
    model.layer1 = resnet18.layer1
    model.layer2 = resnet18.layer2
    model.layer3 = resnet18.layer3
    model.layer4 = resnet18.layer4
    model.avgpool = resnet18.avgpool

    model=model.to(device)
    #optimizer
    optimizer=optim.Adam(model.parameters(), lr=i)
    list.append(train(ep=11,accuracyEveryNthEpoch=2))
    # print(list)
    
    


epoch=12
plt.figure(figsize=(15,11)) 
# labels=["RMSprop (lr=0.5)", "SRMSprop (lr=0.1)", "RMSprop (lr=0.01)", "RMSprop (lr=0.001)"]
labels=["Adam (lr=0.01)","Adam (lr=0.001)","Adam (lr=0.0001)","Adam (lr=0.00001)"]
#training accuracy plot of different activation functions
for i in range(len(labels)):
    axis=plt.subplot(2,2,i+1)
    plt.title(labels[i])
    plt.ticklabel_format(axis='x',useOffset=False)
    plt.plot(range(1,epoch,2),list[i][0], label='Training')
    plt.plot(range(1,epoch,2),list[i][1], label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy%')
    axis.legend(loc='lower right')
plt.savefig("accuracies_activation_functions.png")
