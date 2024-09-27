import os
import numpy as np
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import  DataLoader
import torch.nn as nn 
import json
from torch.utils.data import SubsetRandomSampler, DataLoader
from src.dataset import GRSSDataset, DelhiDataset
from src.lossFunctions import lossDictionary
# from src.model2 import UNet
from src.model_windowedAttention import UNet
# from src.weightInit import he_init, xavier_init
from src.helper import load_checkpoint, ckeck_condition_and_save_state, zoomSelector
import sys
import argparse
from tqdm import tqdm
from src.lossBalancing import LossBalancer

# Define a custom exception
class illigalConfiguration(Exception):
    pass


                                                        #GLOBAL PARAMS

parser = argparse.ArgumentParser(description="Training Script")
# Add argument for config file
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
configPath = parser.parse_args().config

def load_args_from_json(filename=configPath):
    with open(filename, 'r') as f:
        args = json.load(f)
    return argparse.Namespace(**args)
#global parameters
globalParams = load_args_from_json()
    
    
    
    
                                                        #TRAINING DATASET


#parameters for height dataset
datasetParams = globalParams.dataset
loaderParams = datasetParams['loader']

# The output that are to be produced by different decoders. Dataset class will also generate GT based on this produce dictionary
produce=dict()
for i in globalParams.model['decoder']['tracks']:
    produce[i['tag']] = i['produce'] 


# Create separate datasets for GRSS and Delhi
GRSS_dataset = GRSSDataset(datasetParams, produce)
Delhi_dataset = DelhiDataset(datasetParams, produce)

# Set seed for reproducibility
seed = loaderParams['seed']
np.random.seed(seed)
torch.manual_seed(seed)

# Combine datasets for joint training (Optional)
combined_dataset = torch.utils.data.ConcatDataset([GRSS_dataset, Delhi_dataset])

# Dataset size and indices
dataset_size = len(combined_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

# Calculate the split sizes
train_split = int(np.floor(loaderParams['split']['train']  * dataset_size))
val_split = int(np.floor(loaderParams['split']['val']  * dataset_size))

# Create indices for each split
train_indices = indices[:train_split]
val_indices = indices[train_split:train_split + val_split]

# Create samplers for each subset
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create loaders for training and validation
train_loader = DataLoader(combined_dataset, batch_size=loaderParams['batch_size'], sampler=train_sampler, shuffle=False)
val_loader = DataLoader(combined_dataset, batch_size=1, sampler=val_sampler, shuffle=False)

print('height dataset train:val split')
print(len(train_loader)*loaderParams['batch_size'], len(val_loader))
    
trainingParams=globalParams.training   
device=trainingParams['device']

    
    
#                                                        MODEL    
#parameters for model
modelParams = globalParams.model    
    
#deciding on the name of the saved model
modelName='originalGRSS_'
if(datasetParams['include_seam_carving']):
    modelName+='seamCarving_'
modelName=f'DATASET=({modelName[:-1]})__'
restName=''
if(trainingParams['dynamic_loss_balancing']):
    restName+='lossBalancing_' 
for i in modelParams['decoder']['tracks']:
    restName+=f"{i['tag']}({i['produce']})_"
modelName=f'{modelName}DOMAIN_ADAPT=({restName[:-1]})'
print(modelName)

#Defining the model
encoderParams = modelParams['encoder']
decoderParams = modelParams['decoder']
model = UNet(encoderParams, decoderParams)
best_vloss = np.inf
#sending model to cuda device
model = model.to(device)
#optimizer
optimizer = optim.AdamW(model.parameters(), lr=trainingParams['optimizer']['learning_rate'])
# ##initializing the weights via a weights initialization technique in case no pretrained model has been used, 'he' for Relu network and 'xavier' for sigmoid networks 
# # model.apply(he_init)
# # model.apply(xavier_init)

# Deciding on loss Functions
lossFuncMeta = trainingParams['loss_function']
lossFunctions=produce.copy()
for task, i in lossFunctions.items():
    lossFunctions[task] = {j:lossDictionary[lossFuncMeta[j]]() for j in i}

priority = trainingParams['task_priority']
scale_training = trainingParams['multi_scale_training']
lossBalancing = trainingParams['dynamic_loss_balancing']
epoch = trainingParams['epochs']
if(lossBalancing):
    lossBalancer = LossBalancer(priority)
#Tracking the Training losses for loss balancing
trainingLoss = priority.copy()
trainingLoss = {i:[]for i in trainingLoss.keys()}
#Tracking the validation losses for logging
validationLoss = trainingLoss.copy()
zoomSelector = zoomSelector()    

#loading the state of the previously trained model
if(trainingParams["load_checkpoint_file"] != ''):
    checkpoint_file_path = os.path.join(trainingParams["checkpoint_folder"] ,trainingParams["load_checkpoint_file"] )
    logs_file_path = os.path.join(trainingParams["logs_folder"],f'{trainingParams["load_checkpoint_file"]}')
    #loading the variables 
    model, optimizer, best_vloss, logs = load_checkpoint(model, optimizer, checkpoint_file_path, logs_file_path)
    #loading the trainig losses as they will be used in the iterations for calculating dynamic task weights and difficulty
    trainingLoss = logs['trainingLoss']
    validationLoss = logs['validationLoss']
    if(lossBalancing):
        lossBalancer.weights = logs['weights']
        lossBalancer.difficulty = logs['difficulty']
        lossBalancer.AVGLoss = logs['AVGLoss']
    print(f'CHECKPOINT MODEL LOADED: "{checkpoint_file_path}"')


# startEpoch = len(list_avg_loss)






#                                                     TRAINING & VALIDATION

def validation():
    runningTLoss = priority.copy()
    runningTLoss = {i:0 for i in runningTLoss.keys()}
    c=0
    for rgb, GTS, dsmFlag in tqdm(val_loader):
        c+=1
        #input to GPU\
        rgb = rgb.to(device)
        dsmFlag = dsmFlag.to(device)
        #Ground Truth to GPU
        GT = produce.copy()
        for i in GT.keys():
            GT[i] = {j: torch.zeros(1) for j in GT[i]}

        d=0
        for task, j in GT.items():
            for folder in j.keys():
                GT[task][folder] = GTS[:, d:d+1, :, :].to(device)
                d += 1
                
        pred = model(rgb)
        #calculation losses for individual tasks
        individualLoss = computeLoss(pred, GT, lossFunctions, dsmFlag)
        
        #Storing the total running training loss
        for task in individualLoss.keys():
            runningTLoss[task]+=individualLoss[task].item()*priority[task]     
    #Loop ends here
    
    combinedLoss=0    
    for task in runningTLoss.keys():
        if(task == 'dsm'):
            runningTLoss[task]/=c/2
        else:
            runningTLoss[task]/=c
        combinedLoss+=runningTLoss[task]
        
    return (combinedLoss, runningTLoss)
    

def scaleValidations():
    equally_spaced_integers = np.linspace(0, 1, 5)
    for zoom in equally_spaced_integers:
        print(f'zoom: {np.round(1.5*zoom+0.5, 1)}')
        dataset.batchZoom = np.round(1.5*zoom+0.5, 1)
        validation()
                

def computeLoss(pred, GT, lossFn, dsmFlag):
    individualLoss = dict()
    for decoder in pred.keys():
        for task in pred[decoder].keys():
            if(task == 'dsm'):
                dsmFlag = dsmFlag.reshape(dsmFlag.shape[0],1,1,1)
                individualLoss[task] = lossFn[decoder][task](pred[decoder][task]*dsmFlag, GT[decoder][task])
            else:
                individualLoss[task] = lossFn[decoder][task](pred[decoder][task], GT[decoder][task])
    return individualLoss



for i in range(epoch):
    print(f'epoch: {i}')
    runningTLoss = priority.copy()
    runningTLoss = {i:0 for i in runningTLoss.keys()}
    
    c=0
    for rgb, GTS, dsmFlag in tqdm(train_loader):
        c+=1
        # scaling randomly if scaled training is set
        if(scale_training):
            # setting the zoom level for the batch (y = 1.5x + 0.5)
            dataset.batchZoom = np.round(1.5*zoomSelector()+0.5, 1)

        #input to GPU
        rgb = rgb.to(device)
        dsmFlag = dsmFlag.to(device)
        #Ground Truth to GPU
        GT = produce.copy()
        for i in GT.keys():
            GT[i] = {j: torch.zeros(1) for j in GT[i]}

        d=0
        for task, j in GT.items():
            for folder in j.keys():
                GT[task][folder] = GTS[:, d:d+1, :, :].to(device)
                d += 1
                
        #passing the input through the path1
        pred = model(rgb)
        #calculation losses for individual tasks
        individualLoss = computeLoss(pred, GT, lossFunctions, dsmFlag)
#         print([i.item() for i in individualLoss.values()])
        #Storing the total running training loss
        for task in individualLoss.keys():
            runningTLoss[task]+=individualLoss[task].item()*priority[task]
            
        #computing the combined loss from individual losses to backpropagate
        if(lossBalancing):
            combinedLoss = lossBalancer.computeLoss(individualLoss)
        else:
            combinedLoss = 0
            for task in individualLoss.keys():
                combinedLoss+=individualLoss[task]*priority[task]

        #optimizing the weights
        optimizer.zero_grad()
        combinedLoss.backward()
        optimizer.step()

        
        
    for task in runningTLoss.keys():
        trainingLoss[task].append(runningTLoss[task])
    trainingLoss['dsm'][-1]*=2 # This is because loss for dsm is only computed half the time
    
    
    #calculating the validations loss
    model.eval()
    print('validations')
    if(scale_training):
         combinedVLoss, IndividualVLoss = scaleValidations()
    combinedVLoss, IndividualVLoss = validation()
    model.train()
    print(combinedVLoss)
    print(f'TASKS: {list(IndividualVLoss.keys())}')
    print(f'individualVLoss: {list(IndividualVLoss.values())}')
    
    #updating task weights and difficulty
    if(lossBalancing):
        lossBalancer.update(trainingLoss)
        print(f'WEIGHTS: {list(lossBalancer.weights.values())}')
        print(f'Difficulty:{list(lossBalancer.difficulty.values())}')
    
    logs = {'trainingLoss': trainingLoss, 'validationLoss': validationLoss}
    if(lossBalancing):
        logs['AVGLoss']= lossBalancer.AVGLoss
        logs['weights']= lossBalancer.weights
        logs['difficulty']= lossBalancer.weights
    
    best_vloss = ckeck_condition_and_save_state(combinedVLoss, best_vloss, model, modelName, optimizer,
                logs, trainingParams["checkpoint_folder"], trainingParams["logs_folder"]) 

