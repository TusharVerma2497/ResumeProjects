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


google_transform = transforms.Compose([
    transforms.ToTensor(),
])

LISS4_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((53,53))
])

class DelhiDataset(data.Dataset):
    def __init__(self, google_dir, Liss4_dir):
        super(DelhiDataset, self).__init__()
        
        self.google_scale_factor = 1.445/0.6 # downsampling goolge images to match the spectral resolution of 4x upsampled LISS4
        self.google_downsample_shape = (int(512 / self.google_scale_factor), int(512 / self.google_scale_factor))
        
        #Damaged images will be ignored
        ignoreList = []
        with open('ignoreList.pkl', 'rb') as file:
            ignoreList = pickle.load(file)
        lis = list(set(os.listdir(os.path.join(google_dir,'rgb')))-set(ignoreList))
        
        #saving the path to input LISS bands and GT google optical image
        self.google_rgb_images = [os.path.join(google_dir, 'rgb', x) for x in lis]
        self.google_footprint_images = [os.path.join(google_dir, 'footprint', x) for x in lis]
        self.Liss4_B2_images = [os.path.join(Liss4_dir, 'BAND2', x) for x in lis]
        self.Liss4_B3_images = [os.path.join(Liss4_dir, 'BAND3', x) for x in lis]
        self.Liss4_B4_images = [os.path.join(Liss4_dir, 'BAND4', x) for x in lis]
        
        

    def __getitem__(self, index):
        #loading google rgb and building footprint
        hr_google_rgb = np.array(Image.open(self.google_rgb_images[index]))
        hr_google_footprint = np.array(Image.open(self.google_footprint_images[index]))
        
        # downsampling goolge images to match the spectral resolution of 4x upsampled LISS4 as Ground Truth
        hr_google_rgb = cv2.resize(hr_google_rgb[:-40, :-20, :], (self.google_downsample_shape[0], self.google_downsample_shape[1]),interpolation=cv2.INTER_AREA)
        hr_google_footprint = cv2.resize(hr_google_footprint[:-40, :-20], (self.google_downsample_shape[0], self.google_downsample_shape[1]),interpolation=cv2.INTER_AREA)
        hr_google_footprint = applyMorphologicalOpening(hr_google_footprint>0)
        
        #Loading LISS4 bands
        lr_LISS4_B2 = np.array(Image.open(self.Liss4_B2_images[index]), dtype=np.float32)
        lr_LISS4_B3 = np.array(Image.open(self.Liss4_B3_images[index]), dtype=np.float32)
        lr_LISS4_B4 = np.array(Image.open(self.Liss4_B4_images[index]), dtype=np.float32)
        lr_LISS4_Composite = np.stack([lr_LISS4_B2[3:, :], lr_LISS4_B3[3:, :], lr_LISS4_B4[3:, :]], axis = -1).astype(np.float32)
        #To tensor
        hr_google_rgb = google_transform(hr_google_rgb)
        hr_google_footprint = google_transform(hr_google_footprint)
        lr_LISS4_Composite = LISS4_transform(lr_LISS4_Composite)
        
        return lr_LISS4_Composite, hr_google_rgb, hr_google_footprint

    def __len__(self):
        return len(self.google_rgb_images)
    












# import rasterio
# import numpy as np
# from rasterio.transform import rowcol


# class rasterS2_RGB:
#     def __init__(self, image_path, minPercentile = 0.015, maxPercentile = 99.5, transformer = None):
#         self.image_path = image_path
#         # parameters for percentile min-max normalization
#         self.minPercentile = minPercentile
#         self.maxPercentile = maxPercentile
#         # for transforming coordinates to pixel information
#         self.transformer = transformer
#         self.normalizedRGB = self.percentileMinMaxScaling()
#         self.shape = (self.normalizedRGB.shape[1], self.normalizedRGB.shape[2])
# #         self.x_min_pix, self.y_min_pix, self.x_max_pix, self.y_max_pix = self.pixelRangeOfRaster()

    
#     def percentileMinMaxScaling(self):
#         #For percentile normalization of the Raster Image (RGB)
#         def normalize_band(band, min_val, max_val):
#             return np.clip((band - min_val) / (max_val - min_val), 0, 1)
        
#         with rasterio.open(self.image_path) as src:
#             self.transformer = src.transform
#             #Reading RGB bands
#             rasterRGB =  src.read([3, 2, 1])
#             #For computing the percentile min, max of a RGB bands
#             min_r, max_r = np.nanpercentile(rasterRGB[0], self.minPercentile), np.nanpercentile(rasterRGB[0], self.maxPercentile)
#             min_g, max_g = np.nanpercentile(rasterRGB[1], self.minPercentile), np.nanpercentile(rasterRGB[1], self.maxPercentile)
#             min_b, max_b = np.nanpercentile(rasterRGB[2], self.minPercentile), np.nanpercentile(rasterRGB[2], self.maxPercentile)
#             #normalizing
#             rasterRGB[0] = normalize_band(rasterRGB[0], min_r, max_r)
#             rasterRGB[1] = normalize_band(rasterRGB[1], min_g, max_g)
#             rasterRGB[2] = normalize_band(rasterRGB[2], min_b, max_b)
#             return rasterRGB
    
    
#     def pixelRangeOfRaster(self):
#         with rasterio.open(self.image_path) as src:
#             bounds = src.bounds
#             x_min, y_min = self.transformer.transform(bounds.left, bounds.bottom)
#             x_max, y_max = self.transformer.transform(bounds.right, bounds.top)
#             return x_min, y_min, x_max, y_max
        
        
    
    
#     def getPatch(self, TL_lon_lat, BR_lon_lat):
#         # Below function checks if the patch is inside the raster bounds
#         def isPatchInsideRasterBounds(pixelBound):
#             if(pixelBound[0] >= 0 and pixelBound[0] <= self.shape[0] and pixelBound[1] >= 0 and pixelBound[1] <= self.shape[1]):
#                 return True
#             return False
            
#         TL_pixelBound = rowcol(self.transformer, *TL_lon_lat)
#         BR_pixelBound = rowcol(self.transformer, *BR_lon_lat)
# #         print(TL_pixelBound, BR_pixelBound)
#         #if the patch is inside the raster return the patch
#         if(isPatchInsideRasterBounds(TL_pixelBound) and isPatchInsideRasterBounds(BR_pixelBound)):
#             return self.normalizedRGB[:, TL_pixelBound[0]:BR_pixelBound[0]+1, TL_pixelBound[1]:BR_pixelBound[1]+1]
#         #if patch is outside the raster return None
#         else:
#             return None




# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# sentinelTransform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((32, 32))
# ])

# class DelhiDataset(data.Dataset):
#     def __init__(self, dataset_dir, sentinel2Raster_path):
#         super(DelhiDataset, self).__init__()
#         #Loading the sentinel2 Raster
#         self.sentinel2_img = rasterS2_RGB(sentinel2Raster_path, maxPercentile=99.9)
#         #This pattern is used to extract the coordinates from the file name of the google images
#         self.pattern = r"TL_([\d\.\-]+)_([\d\.\-]+)_BR_([\d\.\-]+)_([\d\.\-]+).tif"
        
#         self.scale_factor = 16
#         self.dataset_dir = dataset_dir
#         # Damaged images or images for which equivalent sentinel2 image is not present are ignored
#         ignoreList = []
#         with open('ignoreListSentinelGooglePair.pkl', 'rb') as file:
#             ignoreList = pickle.load(file)
            
#         self.image_filenames = list(set(os.listdir(os.path.join(dataset_dir,'rgb')))-set(ignoreList))
#         self.mr_shape = (int(512 / self.scale_factor*4), int(512 / self.scale_factor*4))
# #         self.lr_shape = (int(512 / self.scale_factor), int(512 / self.scale_factor))


#     def __getitem__(self, index):
#         # Loading the high resolution google ground truth image and synthesizing middle resolution with interpolation
#         hr_image = np.array(Image.open(os.path.join(self.dataset_dir, 'rgb', self.image_filenames[index])))
#         footprint = np.array(Image.open(os.path.join(self.dataset_dir,'footprint', self.image_filenames[index])))
#         mr_image = cv2.resize(hr_image, (self.mr_shape[0], self.mr_shape[1]),interpolation=cv2.INTER_AREA)
#         mr_footprint = cv2.resize(footprint, (self.mr_shape[0], self.mr_shape[1]),interpolation=cv2.INTER_AREA)
#         mr_footprint= (mr_footprint > 2)
# #         lr_image_google= cv2.resize(hr_image, (self.lr_shape[0], self.lr_shape[1]),interpolation=cv2.INTER_AREA)

#         #Loading the low resolution sentinel2 image (This will act as input to the model)
#         match = re.search(self.pattern, self.image_filenames[index])
#         TL_lon_lat = (float(match.group(2)), float(match.group(1)))
#         BR_lon_lat = (float(match.group(4)), float(match.group(3)))
#         lr_image = self.sentinel2_img.getPatch(TL_lon_lat, BR_lon_lat)
#         lr_image = np.transpose(lr_image, (1,2,0))
        
#         #To tensor
#         hr_image = transform(hr_image)
#         mr_image = transform(mr_image)
# #         lr_image_google = transform(lr_image_google)
#         mr_footprint = transform(mr_footprint)
#         lr_image = sentinelTransform(lr_image)
        
#         return lr_image, mr_image, hr_image, mr_footprint

#     def __len__(self):
#         return len(self.image_filenames)