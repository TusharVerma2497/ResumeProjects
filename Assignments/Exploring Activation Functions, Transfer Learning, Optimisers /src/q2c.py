import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import transforms
import numpy as np
import torchvision.datasets as datasets
import torchvision.models as models
from re import TEMPLATE
import matplotlib.pyplot as plt
import itertools
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
                print(f'loss: {loss.item()}')
        print(f'epoch {epoch+1} Finished\n\n')
        if(epoch%accuracyEveryNthEpoch==0):
            train_accuracies.append(accuracy(train_loader))
            test_accuracies.append(accuracy(test_loader))
    return(train_accuracies, test_accuracies)



# Define a set of transformations to be applied to the dataset
transform = transforms.Compose([
    transforms.RandomAffine(degrees=(20, 20)),
    # transforms.ColorJitter(brightness=.5, hue=.3),
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
#     transforms.ElasticTransform(alpha=50.0),
    # transforms.RandomInvert(),
    transforms.RandomSolarize(threshold=92.0),
    transforms.ToTensor(), # Convert the PIL Image to a PyTorch Tensor
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Normalize the image pixel values
])

batchSize=8


train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_data = []
train_targets = []
# test_data = []
# test_targets = []

for i in range(10):
    idx = np.where(np.array(train_set.targets) == i)[0][:500]
    train_data.extend(train_set.data[idx])
    train_targets.extend(np.array(train_set.targets)[idx])
    # idx = np.where(np.array(test_set.targets) == i)[0][:100]
    # test_data.extend(test_set.data[idx])
    # test_targets.extend(np.array(test_set.targets)[idx])

train_set.data = train_data
train_set.targets = train_targets
# test_set.data = test_data
# test_set.targets = test_targets


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize, shuffle=True)





class ResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=[0.2, 0.2, 0.2, 0.2]):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.activations=[]
        
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
        
        # Modify the ResNet18 architecture to include dropout
        self.layer1.add_module("dropout", nn.Dropout(p=dropout_prob[0]))
        self.layer2.add_module("dropout", nn.Dropout(p=dropout_prob[1]))
        self.layer3.add_module("dropout", nn.Dropout(p=dropout_prob[2]))
        self.layer4.add_module("dropout", nn.Dropout(p=dropout_prob[3]))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        self.activations=[]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out) + out
        self.activations.append(out)
        
        temp= self.layer2(out)
        # padding is added to make dimensions compatible
        out = F.pad(out[:,:,::2,::2], (0,0,0,0,out.shape[1]//2, out.shape[1]//2,0,0))
        out=temp+out
        self.activations.append(out)
        
        temp= self.layer3(out)
        out = F.pad(out[:,:,::2,::2], (0,0,0,0,out.shape[1]//2, out.shape[1]//2,0,0))
        out=temp+out
        self.activations.append(out)
        
        temp= self.layer4(out)
        out = F.pad(out[:,:,::2,::2], (0,0,0,0,out.shape[1]//2, out.shape[1]//2,0,0))
        out=temp+out
        self.activations.append(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out



def savePlots(index, plots):
    for j in range(10):
        labels=["input tensor","after conv1 block", "after conv2 block", "after conv3 block","after conv4 block"]
        for i in range(5):
            axi=plt.subplot(2,3,i+1)
            plt.title(labels[i])
            axi.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
            axi.imshow(plots[j][i])
        plt.savefig('img'+str(index)+str(j)+".png")


def displayActivations(index, loader=test_loader):
    plots=[]
    for i, (img,target) in enumerate(loader):
        img=img.to(device)
        out=model(img)
        predicted_digit=torch.argmax(out, dim=1)
        for j,k in enumerate(zip(predicted_digit, target)):
            if(k[0]==k[1]):
                sp=[]
                image=img[j]
                image=image.cpu()
                image = image.permute(1, 2, 0).detach().numpy()
                sp.append(image)
                for l in range(4):
#                     axi=plt.subplot(2,2,l+1)
#                     axi.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
                    image=model.activations[l][j][2]
                    image=image.unsqueeze(0)
                    image=image.cpu()
                    image = image.permute(1, 2, 0).detach().numpy()
#                     axi.imshow(image)
#                     print(l)
                    sp.append(image)
                plots.append(sp)
        if(len(plots)>=10):
            savePlots(index, plots)
            break


dropouts=[[0.5, 0.5, 0.3, 0.1],[0.5, 0.4, 0.3, 0.2],[0.1, 0.2, 0.3, 0.4],[0.3, 0.3, 0.3, 0.3],[0.6, 0.6, 0.6, 0.6]]
lossFunction=nn.CrossEntropyLoss()
list=[]
for i in range(len(dropouts)):
    model = ResNet18(dropout_prob=dropouts[i])
    model=model.to(device)
    #optimizer
    optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.2)
    list.append(train(ep=41,accuracyEveryNthEpoch=3))
    displayActivations(i)



