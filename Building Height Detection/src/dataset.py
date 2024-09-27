# import os
# import numpy as np
# from tqdm import tqdm
# from torch.nn import functional as F
# from torch.utils import data
# from torch.utils.data import  DataLoader
# import torch.nn as nn 
# from torchvision import transforms
# import json
# import random
# from PIL import Image
# from src.helper import applyMorphologicalOpening, dynamicZoom
# import torch
# import matplotlib.pyplot as plt                         


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((128, 128), antialias=True, interpolation=3),
#     transforms.RandomHorizontalFlip(p=0.65),
#     transforms.RandomVerticalFlip(p=0.65),
# ])

# GRSS_normalize = transforms.Compose([
#     transforms.Normalize(mean=[81.1724, 87.8123, 71.9168], std=[41.2546, 37.0713, 37.4346]),
# ])

# Delhi_normalize = transforms.Compose([
#     transforms.Normalize(mean=[101.4604,  99.9694, 105.2648], std=[53.8622, 49.4224, 45.1765])
# ])



# class GRSSDataset(data.Dataset):
#     def __init__(self, params, produce):   
#         GRSS_rootList = [params['GRSS_rootPath']]
#         GRSS_rootList = GRSS_rootList + params['seamCarving_rootPath'] if params['include_seam_carving'] else GRSS_rootList
        
#         self.batchZoom = 0.8
#         self.GRSS_normalize = GRSS_normalize
#         self.transform = transform
#         self.produce = produce
#         self.all_possible_produce = {j for i in produce.values() for j in i}
        
# #         self.GRSS_GT_path = {i: [] for i in self.all_possible_produce - {'footprint'}}
#         self.GRSS_GT_path = {i: [] for i in self.all_possible_produce}
#         self.GRSS_input = []
#         self.colorJitterFlag = params['color_jitter']
#         self.colorJitter = transforms.ColorJitter(brightness=(0.5, 1.3), contrast=(0.6, 2), saturation=(0.6, 2))
        
#         GRSS_fileList = os.listdir(os.path.join(GRSS_rootList[0], 'dsm'))
#         for i in GRSS_rootList:
#             for j in GRSS_fileList:
#                 self.GRSS_input.append(os.path.join(i, 'rgb', j))
#                 for tag in self.GRSS_GT_path.keys():
#                     if( tag == 'footprint'):
#                         self.GRSS_GT_path[tag].append(os.path.join(i, 'dsm', j))
#                     else:
#                         self.GRSS_GT_path[tag].append(os.path.join(i, tag, j))
    
#     def __len__(self):
#         return len(self.GRSS_input)
    
#     def __getitem__(self, index):
#         dsmFlag = torch.ones(1)
#         GT = self.produce
# #         for i in GT.keys():
# #             GT[i] = {j: torch.zeros(1) for j in GT[i]}
            
# #         print(self.GRSS_input[index])
#         rgb = np.array(Image.open(self.GRSS_input[index]))
#         inp_out = [rgb]
        

#         for task, j in GT.items():
#             for folder in j:
#                 if folder == 'footprint':
#                     temp = np.array(Image.open(self.GRSS_GT_path[folder][index]))
#                     temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
#                     temp = applyMorphologicalOpening(temp > 0)
#                 else:
#                     temp = np.array(Image.open(self.GRSS_GT_path[folder][index]))
#                     temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
#                 inp_out.append(temp)

#         combined = np.concatenate(inp_out, axis=2)
        
#         if '100SeamRemoved' in self.GRSS_input[index]:
#             combined = dynamicZoom(combined, 0.804)
#         elif '150SeamRemoved' in self.GRSS_input[index]:
#             combined = dynamicZoom(combined, 0.706)

# #         combined = dynamicZoom(combined, self.batchZoom)
#         combined = combined.astype(np.float32)
#         combined = self.transform(combined)
#         rgb = combined[:3, :, :]
#         rgb = self.GRSS_normalize(rgb)
#         GT = combined[3:, :, :]
# #         c = 3
# #         for task, j in GT.items():
# #             for folder in j.keys():
# #                 GT[task][folder] = combined[c:c+1, :, :]
# #                 c += 1

#         if self.colorJitterFlag:
#             rgb = self.colorJitter(rgb)

#         return (rgb, GT, dsmFlag)
    
    
    
    

    
# class DelhiDataset(data.Dataset):
#     def __init__(self, params, produce):
#         Delhi_root = params['Delhi_rootPath']
        
#         Delhi_JSONpath = params['Delhi_json']
#         with open(Delhi_JSONpath, 'r') as f:
#             info = json.load(f)
#         Delhi_fileList = [k for k in info.keys()][:params['num_delhi_files_for_training']]
        
#         self.batchZoom = 1
#         self.Delhi_normalize = Delhi_normalize
#         self.transform = transform
#         self.produce = produce
#         self.all_possible_produce = {j for i in produce.values() for j in i}
        
#         self.Delhi_GT_path = {i: [] for i in self.all_possible_produce - {'dsm'}}
#         self.Delhi_input = []
#         self.colorJitterFlag = params['color_jitter']
#         self.colorJitter = transforms.ColorJitter(brightness=(0.5, 1.3), contrast=(0.6, 2), saturation=(0.6, 2))
        
#         for j in Delhi_fileList:
#             self.Delhi_input.append(os.path.join(Delhi_root, 'rgb', j))
#             for tag in self.Delhi_GT_path.keys():
#                 self.Delhi_GT_path[tag].append(os.path.join(Delhi_root, tag, j))
    
#     def __len__(self):
#         return len(self.Delhi_input)
    
#     def __getitem__(self, index):
#         GT = self.produce
#         dsmFlag = torch.ones(1)
# #         for i in GT.keys():
# #             GT[i] = {j: torch.zeros(1) for j in GT[i]}
        
#         rgb = np.array(Image.open(self.Delhi_input[index]))
#         inp_out = [rgb]
        
#         for task, j in GT.items():
#             for folder in j:
#                 if folder == 'dsm':
#                     temp = np.zeros((512,512,1))
#                     dsmFlag = torch.zeros(1)
#                 else:
#                     temp = np.array(Image.open(self.Delhi_GT_path[folder][index]))
#                     temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
#                     if folder == 'footprint':
#                         temp = applyMorphologicalOpening(temp > 0)
#                 inp_out.append(temp)
                
#         combined = np.concatenate(inp_out, axis=2)
        
#         combined = dynamicZoom(combined, self.batchZoom)
#         combined = combined.astype(np.float32)
#         combined = self.transform(combined)
#         rgb = combined[:3, :, :]
#         rgb = self.Delhi_normalize(rgb)
#         GT = combined[3:, :, :]
# #         c = 3
# #         for task, j in GT.items():
# #             for folder in j.keys():
# #                 GT[task][folder] = combined[c:c+1, :, :]
# #                 c += 1

#         if self.colorJitterFlag:
#             rgb = self.colorJitter(rgb)

#         return (rgb, GT, dsmFlag)











import os
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import  DataLoader
import torch.nn as nn 
from torchvision import transforms
import json
import random
from PIL import Image
from src.helper import applyMorphologicalOpening, dynamicZoom
import torch
import matplotlib.pyplot as plt  
import cv2


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128), antialias=True, interpolation=3),
    transforms.RandomHorizontalFlip(p=0.65),
    transforms.RandomVerticalFlip(p=0.65),
])

GRSS_normalize = transforms.Compose([
    transforms.Normalize(mean=[81.1724, 87.8123, 71.9168], std=[41.2546, 37.0713, 37.4346]),
])

Delhi_normalize = transforms.Compose([
    transforms.Normalize(mean=[101.4604,  99.9694, 105.2648], std=[53.8622, 49.4224, 45.1765])
])



class GRSSDataset(data.Dataset):
    def __init__(self, params, produce):   
        GRSS_rootList = [params['GRSS_rootPath']]
        GRSS_rootList = GRSS_rootList + params['seamCarving_rootPath'] if params['include_seam_carving'] else GRSS_rootList
        
        self.batchZoom = 0.8
        self.GRSS_normalize = GRSS_normalize
        self.transform = transform
        self.produce = produce
        self.all_possible_produce = {j for i in produce.values() for j in i}
        
#         self.GRSS_GT_path = {i: [] for i in self.all_possible_produce - {'footprint'}}
        self.GRSS_GT_path = {i: [] for i in self.all_possible_produce}
        self.GRSS_input = []
        self.colorJitterFlag = params['color_jitter']
        self.colorJitter = transforms.ColorJitter(brightness=(0.5, 1.3), contrast=(0.6, 2), saturation=(0.6, 2))
        
        GRSS_fileList = os.listdir(os.path.join(GRSS_rootList[0], 'dsm'))
        for i in GRSS_rootList:
            for j in GRSS_fileList:
                self.GRSS_input.append(os.path.join(i, 'rgb', j))
                for tag in self.GRSS_GT_path.keys():
                    if( tag == 'refined_shadow'):
                        self.GRSS_GT_path[tag].append((os.path.join(i, 'shadow', j), os.path.join(i, 'vegetation_ndvi', j)))
                    elif( tag == 'footprint'):
                        self.GRSS_GT_path[tag].append(os.path.join(i, 'dsm', j))
                    else:
                        self.GRSS_GT_path[tag].append(os.path.join(i, tag, j))
    
    def __len__(self):
        return len(self.GRSS_input)
    
    def __getitem__(self, index):
        dsmFlag = torch.ones(1)
        GT = self.produce
#         for i in GT.keys():
#             GT[i] = {j: torch.zeros(1) for j in GT[i]}
            
#         print(self.GRSS_input[index])
        rgb = np.array(Image.open(self.GRSS_input[index]))
#         rgb = cv2.bilateralFilter(rgb, 20, 175, 50)
        inp_out = [rgb]
        

        for task, j in GT.items():
            for folder in j:
                if folder == 'footprint':
                    temp = np.array(Image.open(self.GRSS_GT_path[folder][index]))
                    temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
                    temp = applyMorphologicalOpening(temp > 0)
                elif folder == 'refined_shadow':
                    shadow = np.array(Image.open(self.GRSS_GT_path[folder][index][0])) 
                    vegetation = np.array(Image.open(self.GRSS_GT_path[folder][index][1]))
                    refinedShadow = shadow * (~vegetation)
                    temp = refinedShadow.reshape(refinedShadow.shape[0], refinedShadow.shape[1], 1)
                else:
                    temp = np.array(Image.open(self.GRSS_GT_path[folder][index]))
                    temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
                inp_out.append(temp)

        combined = np.concatenate(inp_out, axis=2)
        
        if '100SeamRemoved' in self.GRSS_input[index]:
            combined = dynamicZoom(combined, 0.804)
        elif '150SeamRemoved' in self.GRSS_input[index]:
            combined = dynamicZoom(combined, 0.706)

        combined = dynamicZoom(combined, self.batchZoom)
        combined = combined.astype(np.float32)
        combined = self.transform(combined)
        rgb = combined[:3, :, :]
        if self.colorJitterFlag:
            rgb = self.colorJitter(rgb)
        rgb = self.GRSS_normalize(rgb)
        GT = combined[3:, :, :]
#         c = 3
#         for task, j in GT.items():
#             for folder in j.keys():
#                 GT[task][folder] = combined[c:c+1, :, :]
#                 c += 1

        

        return (rgb, GT, dsmFlag)
    
    
    
    

    
class DelhiDataset(data.Dataset):
    def __init__(self, params, produce):
        Delhi_root = params['Delhi_rootPath']
        
        Delhi_JSONpath = params['Delhi_json']
        with open(Delhi_JSONpath, 'r') as f:
            info = json.load(f)
        Delhi_fileList = [k for k in info.keys()][:params['num_delhi_files_for_training']]
        
        self.batchZoom = 1
        self.Delhi_normalize = Delhi_normalize
        self.transform = transform
        self.produce = produce
        self.all_possible_produce = {j for i in produce.values() for j in i}
        
        self.Delhi_GT_path = {i: [] for i in self.all_possible_produce - {'dsm'}}
        self.Delhi_input = []
        self.colorJitterFlag = params['color_jitter']
        self.colorJitter = transforms.ColorJitter(brightness=(0.5, 1.3), contrast=(0.6, 2), saturation=(0.6, 2))
        
        for j in Delhi_fileList:
            self.Delhi_input.append(os.path.join(Delhi_root, 'rgb', j))
            for tag in self.Delhi_GT_path.keys():
                if( tag == 'refined_shadow'):
                    self.Delhi_GT_path[tag].append((os.path.join(Delhi_root, 'shadow', j), os.path.join(Delhi_root, 'vegetation_ndvi', j)))
                else:
                    self.Delhi_GT_path[tag].append(os.path.join(Delhi_root, tag, j))
    
    def __len__(self):
        return len(self.Delhi_input)
    
    def __getitem__(self, index):
        GT = self.produce
        dsmFlag = torch.ones(1)
#         for i in GT.keys():
#             GT[i] = {j: torch.zeros(1) for j in GT[i]}
        
        rgb = np.array(Image.open(self.Delhi_input[index]))
#         rgb = cv2.bilateralFilter(rgb, 20, 175, 50)
        inp_out = [rgb]
        
        for task, j in GT.items():
            for folder in j:
                if folder == 'dsm':
                    temp = np.zeros((512,512,1))
                    dsmFlag = torch.zeros(1)
                elif folder == 'refined_shadow':
                    shadow = np.array(Image.open(self.Delhi_GT_path[folder][index][0])) 
                    vegetation = np.array(Image.open(self.Delhi_GT_path[folder][index][1]))
                    refinedShadow = shadow * (~vegetation)
                    temp = refinedShadow.reshape(refinedShadow.shape[0], refinedShadow.shape[1], 1)
                else:
                    temp = np.array(Image.open(self.Delhi_GT_path[folder][index]))
                    temp = temp.reshape(temp.shape[0], temp.shape[1], 1)
                    if folder == 'footprint':
                        temp = applyMorphologicalOpening(temp > 0)
                inp_out.append(temp)
                
        combined = np.concatenate(inp_out, axis=2)
        
        combined = dynamicZoom(combined, self.batchZoom)
        combined = combined.astype(np.float32)
        combined = self.transform(combined)
        rgb = combined[:3, :, :]
        if self.colorJitterFlag:
            rgb = self.colorJitter(rgb)
        rgb = self.Delhi_normalize(rgb)
        GT = combined[3:, :, :]
#         c = 3
#         for task, j in GT.items():
#             for folder in j.keys():
#                 GT[task][folder] = combined[c:c+1, :, :]
#                 c += 1

        

        return (rgb, GT, dsmFlag)
