from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image
import cv2
import pickle
import re
import scipy.ndimage as ndi


# morphological opening for removing unwanted artifacts form footprint images
def applyMorphologicalOpening(mask, iterations=3):
#     mask = mask[:,:]
    cleaned_mask = ndi.binary_opening(mask, iterations=iterations)
    cleaned_mask = cleaned_mask.reshape(mask.shape[0],mask.shape[1],1)
    return cleaned_mask


HighRes_transform = transforms.Compose([
    transforms.ToTensor(),
])

LISS4_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((53,53))
])

class DelhiDataset(data.Dataset):
    def __init__(self, hr_image_dir, Liss4_dir, hr_downscale_factor = 1.445/0.6):
        super(DelhiDataset, self).__init__()
        
        self.hr_downscale_factor = hr_downscale_factor # downsampling goolge images to match the spectral resolution of 4x upsampled LISS4
        self.hr_downsample_shape = (int(512 / self.hr_downscale_factor), int(512 / self.hr_downscale_factor))
        
        #Damaged images will be ignored
        # ignoreList = []
        # with open('ignoreList.pkl', 'rb') as file:
        #     ignoreList = pickle.load(file)
        # lis = list(set(os.listdir(os.path.join(hr_image_dir,'rgb')))-set(ignoreList))
        
        #saving the path to input LISS bands and GT High Resolution optical image
        self.hr_rgb_images = [os.path.join(hr_image_dir, 'rgb', x) for x in lis]
        self.hr_footprint_images = [os.path.join(hr_image_dir, 'footprint', x) for x in lis]
        self.Liss4_B2_images = [os.path.join(Liss4_dir, 'BAND2', x) for x in lis]
        self.Liss4_B3_images = [os.path.join(Liss4_dir, 'BAND3', x) for x in lis]
        self.Liss4_B4_images = [os.path.join(Liss4_dir, 'BAND4', x) for x in lis]
        
        

    def __getitem__(self, index):
        #loading google rgb and building footprint
        hr_rgb = np.array(Image.open(self.hr_rgb_images[index]))
        hr_footprint = np.array(Image.open(self.hr_footprint_images[index]))
        
        # downsampling High Res images to match the spectral resolution of 4x upsampled LISS4 as Ground Truth
        hr_rgb = cv2.resize(hr_rgb[:-40, :-20, :], (self.hr_downsample_shape[0], self.hr_downsample_shape[1]),interpolation=cv2.INTER_AREA)
        hr_footprint = cv2.resize(hr_footprint[:-40, :-20], (self.hr_downsample_shape[0], self.hr_downsample_shape[1]),interpolation=cv2.INTER_AREA)
        hr_footprint = applyMorphologicalOpening(hr_footprint>0)
        
        #Loading LISS4 bands
        lr_LISS4_B2 = np.array(Image.open(self.Liss4_B2_images[index]), dtype=np.float32)
        lr_LISS4_B3 = np.array(Image.open(self.Liss4_B3_images[index]), dtype=np.float32)
        lr_LISS4_B4 = np.array(Image.open(self.Liss4_B4_images[index]), dtype=np.float32)
        lr_LISS4_Composite = np.stack([lr_LISS4_B2[3:, :], lr_LISS4_B3[3:, :], lr_LISS4_B4[3:, :]], axis = -1).astype(np.float32)
        #To tensor
        hr_rgb = HighRes_transform(hr_rgb)
        hr_footprint = HighRes_transform(hr_footprint)
        lr_LISS4_Composite = LISS4_transform(lr_LISS4_Composite)
        
        return lr_LISS4_Composite, hr_rgb, hr_footprint

    def __len__(self):
        return len(self.hr_rgb_images)
    