
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import torch.nn as nn 
from torchvision import transforms
import scipy.ndimage as ndi
import random
import torch
import pickle
from numpy.polynomial.polynomial import Polynomial
from collections import defaultdict, deque

# morphological opening for removing unwanted artifacts form footprint images
def applyMorphologicalOpening(mask, iterations=3):
    mask = mask[:,:,0]
    cleaned_mask = ndi.binary_opening(mask, iterations=iterations)
    cleaned_mask = cleaned_mask.reshape(mask.shape[0],mask.shape[1],1)
    return cleaned_mask


# zooming range is from 0.5 to 2 where 0.5 meaning zoomout and 2 means zoomIn and 1 is original image
def dynamicZoom(img, zoomLevel):
    
    assert zoomLevel>=0.5 and zoomLevel<=2
    if(len(img.shape)==2):
        img =img.reshape(img.shape[0], img.shape[1],1)
        
    scale = 1/zoomLevel
    
    original_length = img.shape[0]
    original_width = img.shape[1]
    
    new_shape = (int(original_length*scale), int(original_width*scale))
    
    if(zoomLevel>=1):
        return img[:new_shape[0], :new_shape[1]]
    else:
        newImg = np.zeros((new_shape[0],new_shape[1],img.shape[2]), dtype = np.uint8)
        newImg[:original_length, :original_width] = img
        newImg[original_length:,:original_width] = img[:new_shape[0]-original_length,:]
        newImg[:original_length, original_width:] = img[:,:new_shape[0]-original_width]
        newImg[original_length:, original_width:]= img[:new_shape[0]-original_length,:new_shape[0]-original_width]
        return newImg



# for zoom training
class zoomSelector:
    def __init__(self):
#         starting with a uniform distribution
#         self.zoom_levels, self.probabilities  = self.updateSelector(np.linspace(0, 1, 5), [1,1,1,1,1])
        self.zoom_levels = np.linspace(0, 1, 100)
        self.probabilities = [1 / len(self.zoom_levels)] * len(self.zoom_levels)
        
    def __call__(self):    
        sampled_zoom = np.random.choice(self.zoom_levels, p=self.probabilities)
        return sampled_zoom
    
    def update(self, zoom_levels, losses):
        
#         This Function fits a distribution to the looes at particulat zoom level and than 
#         helps in sampling from the zoom level at which loss is more
        def fit_distribution(x_vals, y_vals, degree=3, num_points=100):
            # Fit a polynomial to the provided values
            poly = Polynomial.fit(x_vals, y_vals, degree)
            # Generate continuous x values (e.g., zoom levels)
            x_continuous = np.linspace(min(x_vals), max(x_vals), num_points)
            # Calculate corresponding y values using the polynomial
            y_continuous = poly(x_continuous)
            # Ensure all y values are positive to make them valid probabilities
            y_continuous = np.maximum(y_continuous, 0)
            # Normalize the y values to sum to 1
            y_continuous /= np.sum(y_continuous)
            return x_continuous, y_continuous
        
        self.zoom_levels, self.probabilities  = fit_distribution(zoom_levels, losses)


    
    
    
                                                # SAVING & LOADING MODEL   

# 
    
# loading the state space of the best model optimizer state
def load_checkpoint(model, optimizer, state_path, logs_path):
    model_state = torch.load(state_path)['state_dict']
    optimizer_state = torch.load(state_path)['optimizer']
    best_vloss = torch.load(state_path)['best_vloss']
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    
    with open(logs_path, 'rb') as file:
        logs= pickle.load(file)
    return model, optimizer, best_vloss, logs



# checking condition and saving the checkpoint
def ckeck_condition_and_save_state(avg_vloss, best_vloss, model, modelName, optimizer, logs, state_root, logs_root):
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print("Height Model improved, saving...")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_vloss": best_vloss
        }
        print("=> Saving checkpoint")
        torch.save(checkpoint, os.path.join(state_root,modelName))
        with open(os.path.join(logs_root,modelName), 'wb') as file:
            pickle.dump(logs, file)
#         torch.save(checkpoint, os.path.join(state_root,f"Model_valid_loss:{round(best_vloss,3)}.pth.tar"))
#         with open(os.path.join(logs_root,f"Model_valid_loss:{round(best_vloss,3)}.pth.tar.pkl"), 'wb') as file:
#             pickle.dump(logs, file)
            
    print('\n')
    return best_vloss








# TOPOLOGICAL SORT FOR DETECTING which decoder should be executed first in case of cross_attention


def topological_sort(pairs):
    # Create adjacency list and in-degree count
    adj_list = defaultdict(list)
    in_degree = defaultdict(int)
    
    # Build graph
    nodes = set()
    for a, b in pairs:
        adj_list[a].append(b)
        in_degree[b] += 1
        nodes.add(a)
        nodes.add(b)
    
    # Initialize queue with nodes that have in-degree of 0
    zero_in_degree = deque([node for node in nodes if in_degree[node] == 0])
    
    topological_order = []
    
    while zero_in_degree:
        current = zero_in_degree.popleft()
        topological_order.append(current)
        
        # Process neighbors and reduce their in-degree
        for neighbor in adj_list[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)
    
    # Check if topological sort is possible (i.e., no cycle)
    if len(topological_order) == len(nodes):
        return topological_order
    else:
        return "Cycle detected! Topological sort is not possible."

