import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import numpy as np

#This is working perfectly a differentiable height filter
class HeightFilter(nn.Module):
    def __init__(self, shift=2, alpha=30):
        super().__init__()
        self.shift = shift
        self.alpha=alpha

    def forward(self, x):
        sigmoidFilter = torch.sigmoid(self.alpha*(x - self.shift))
        reluFilter = torch.relu(x- self.shift) + self.shift
        
        return reluFilter*sigmoidFilter



class DiceLoss(nn.Module):
    def __init__(self, smooth=1, footprint_alpha=30, shift=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.footprint_alpha = footprint_alpha
        self.shift=shift

    def forward(self, inputs, targets): # Expect both the inpur to be a binary mask
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth)
        
        return dice_loss

class DiceRMSELoss(nn.Module):
    def __init__(self):
        super(DiceL2Loss, self).__init__()
        self.L2_loss = nn.MSELoss(reduction='mean')
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):

        rmse_loss = torch.sqrt(self.L2_loss(inputs, targets))
        dice_loss = self.dice_loss(inputs, targets)
        #combining both the loss
        Dice_L2 = rmse_loss + dice_loss
        return Dice_L2


class DiceL1Loss(nn.Module):
    def __init__(self):
        super(DiceL1Loss, self).__init__()
        self.L1_loss = nn.SmoothL1Loss(reduction='mean')
        self.dice_loss = DiceLoss()
        self.sigmoid =nn.Sigmoid()
        
    def forward(self, inputs, targets, flag='Height'):

        l1_loss1 = self.L1_loss(inputs, targets)
        if(flag=='Height'):
            dice_loss = self.dice_loss(self.sigmoid(30*inputs), self.sigmoid(30*targets))
        else:
            dice_loss = self.dice_loss(inputs, targets)
        #combining both the loss
        Dice_L1 = l1_loss1 + dice_loss
        return Dice_L1


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce_losss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets, footprint_alpha=30):
        
        dice_loss = self.dice_loss(inputs, targets)
        inputs = torch.sigmoid(footprint_alpha*inputs)
        BCE = self.bce_losss(inputs, targets)
        #combining both the loss
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

    
    
    
class BoundaryBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BoundaryBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss()
#         self.boundary_loss = boundary_loss

    def forward(self, inputs, targets, smooth=1, footprint_alpha=20):
        
        inputs = torch.sigmoid(footprint_alpha*inputs)
        
        #computing BCE loss
        BCE = self.bce_loss(inputs, targets)
        #computing Boundary loss
        Boundary = self.boundary_loss(inputs, targets)
        #combining the losses
        Dice_Boundary_BCE = BCE + Boundary
        return Dice_Boundary_BCE
    



# def boundary_loss(pred, gt):
    
#     def compute_distance_map(mask):
#         mask = mask.detach().cpu().numpy()
#         distance_map = distance_transform_edt(mask) + distance_transform_edt(1 - mask)
#         return torch.tensor(distance_map).float()

#     pred_boundary = compute_distance_map(pred)
#     gt_boundary = compute_distance_map(gt)

#     loss = torch.abs(pred_boundary - gt_boundary).mean()
#     return loss




device = "cuda" if torch.cuda.is_available() else "cpu"

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        n, c, _, _ = pred.shape
        # boundary map
        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred
        
        gt_b = F.max_pool2d(1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt
        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)
        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)
        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


    

# # This function is non differentiable so cannot be used as an objective function
# class bhattacharyyaDistance(nn.Module):
#     def __init__(self):
#         super().__init__()
#         buildingFilter = HeightFilter()

#     def forward(self, pred, tar):
#         # Flatten the inputs along the spatial dimensions
#         pred_flat = buildingFilter(pred.view(pred.size(0), -1)) # [batch, channels*width*height]
#         tar_flat = buildingFilter(tar.view(tar.size(0), -1))  # [batch, channels*width*height]


#         # Calculate the bin edges for the histogram
#         min_val = min(torch.min(pred).item(), torch.min(tar).item())
#         max_val = max(torch.max(pred).item(), torch.max(tar).item())
#         bins= int((max_val-min_val)//2)
#         bin_edges = torch.linspace(min_val, max_val, bins + 1).to(pred.device)
        

#         # Compute the histogram for predictions and target
#         hist_pred = torch.histc(pred_flat, bins=bins, min=min_val, max=max_val)
#         hist_tar = torch.histc(tar_flat, bins=bins, min=min_val, max=max_val)

#         # Normalize histograms to form probability distributions
#         hist_pred = hist_pred / hist_pred.sum(dim=-1, keepdim=True)
#         hist_tar = hist_tar / hist_tar.sum(dim=-1, keepdim=True)

#         # Compute Bhattacharyya coefficient
#         bc = torch.sum(torch.sqrt(hist_pred * hist_tar), dim=-1)

#         # Compute Bhattacharyya distance
#         bhattacharyya_dist = -torch.log(bc + 1e-10)  # Adding epsilon for numerical stability

#         # Return the mean distance over the batch
#         return torch.mean(bhattacharyya_dist)


    

    
    
    
# Possibly differentialbe function    
class BhattacharyyaDistance(nn.Module):
    def __init__(self, sigma=0.2):
        super().__init__()
#         self.bins = bins
        self.sigma = sigma
        self.buildingFilter = HeightFilter()
#         sig = nn.Sigmoid()

    def forward(self, pred, tar):
        pred_flat = self.buildingFilter(pred.view(pred.size(0), -1))  # [batch, channels*width*height]
        tar_flat = self.buildingFilter(tar.view(tar.size(0), -1))  # [batch, channels*width*height]

        min_val = min(torch.min(pred).item(), torch.min(tar).item())
        max_val = max(torch.max(pred).item(), torch.max(tar).item())
        self.bins= int(max_val-min_val)
        # Create the bins for the histogram
        bin_centers = torch.linspace(min_val, max_val, self.bins).to(pred.device)
        
        # Compute the Gaussian-smoothed histograms
        hist_pred = self.gaussian_histogram(pred_flat, bin_centers)
        hist_tar = self.gaussian_histogram(tar_flat, bin_centers)

        # Normalize histograms to form probability distributions
        hist_pred = hist_pred / hist_pred.sum(dim=-1, keepdim=True)
        hist_tar = hist_tar / hist_tar.sum(dim=-1, keepdim=True)

        # Compute Bhattacharyya coefficient
        bc = torch.sum(torch.sqrt(hist_pred * hist_tar), dim=-1)

        # Compute Bhattacharyya distance
        bhattacharyya_dist = -torch.log(bc + 1e-10)  # Adding epsilon for numerical stability

        return torch.mean(bhattacharyya_dist)

    def gaussian_histogram(self, values, bin_centers):
#         sig=nn.Sigmoid()
        # Compute distances to bin centers
        diff = values.unsqueeze(-1) - bin_centers.unsqueeze(0)
        hist = torch.exp(-0.5 * (diff / self.sigma) ** 2)  # Gaussian kernel
#         hist = sig(30*hist)
        # Sum over the flattened spatial dimensions
        hist = hist.sum(dim=1)

        return hist
    
    
    
    
class DiceBhattacharyyaDistanceL1Loss(nn.Module):
    def __init__(self):
        super(DiceBhattacharyyaDistanceL1Loss, self).__init__()
        self.bhattacharyyaLoss = BhattacharyyaDistance()
        self.L1_loss = nn.SmoothL1Loss(reduction='mean')
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets, smooth=1):
        Loss_l1 = self.L1_loss(inputs, targets)
        Bhatt_loss = self.bhattacharyyaLoss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        #combining both the loss
        Dice_Bhatt_l1 = Bhatt_loss + dice_loss + Loss_l1
        return Dice_Bhatt_l1
    

class L1BhattacharyyaDistanceLoss(nn.Module):
    def __init__(self):
        super(L1BhattacharyyaDistanceLoss, self).__init__()
        self.bhattacharyyaLoss = BhattacharyyaDistance()
        self.L1_loss = nn.SmoothL1Loss(reduction='mean')
#         self.dice_loss = DiceLoss()
        

    def forward(self, inputs, targets, smooth=1):
        Loss_l1 = self.L1_loss(inputs, targets)
        Bhatt_loss = self.bhattacharyyaLoss(inputs, targets)
#         dice_loss = self.dice_loss(inputs, targets)
        #combining both the loss
        L1_Bhatt = Bhatt_loss + Loss_l1
        return L1_Bhatt
    
class DiceBhattacharyyaDistanceLoss(nn.Module):
    def __init__(self):
        super(DiceBhattacharyyaDistanceLoss, self).__init__()
        self.bhattacharyyaLoss = BhattacharyyaDistance()
#         self.L1_loss = nn.SmoothL1Loss(reduction='mean')
        self.dice_loss = DiceLoss()
        

    def forward(self, inputs, targets, smooth=1):
#         Loss_l1 = self.L1_loss(inputs, targets)
        Bhatt_loss = self.bhattacharyyaLoss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        #combining both the loss
        Dice_Bhatt = Bhatt_loss + dice_loss
        return Dice_Bhatt
    
    


def sobel_image_gradients(image):
    # Sobel filters for computing gradients in the x and y directions
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)

    # Apply Sobel filters to get gradients
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)

    return grad_x, grad_y

class GradientMatchingLoss(nn.Module):
    def __init__(self):
        super(GradientMatchingLoss, self).__init__()

    def forward(self, pred, target):
#         pred = torch.sigmoid(30 * (pred-2))
        # Compute gradients for both predicted and target images
        grad_x_pred, grad_y_pred = sobel_image_gradients(pred)
        grad_x_target, grad_y_target = sobel_image_gradients(target)
        # Calculate the Mean Squared Error between the gradients
        loss = F.mse_loss(grad_x_pred, grad_x_target) + F.mse_loss(grad_y_pred, grad_y_target)
        return loss

    
class DiceGradientLoss(nn.Module):
    def __init__(self):
        super(DiceGradientLoss, self).__init__()
        self.gradient_loss = GradientMatchingLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        
        dice_loss = self.dice_loss(inputs, targets)
        gradient_loss = self.gradient_loss(inputs, targets)
        #combining both the loss
        Dice_gradient_loss = gradient_loss + dice_loss
        return Dice_gradient_loss
    
    
    
class GradientL1Loss(nn.Module):
    def __init__(self):
        super(GradientL1Loss, self).__init__()
        self.gradient_loss = GradientMatchingLoss()
        self.l1_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, targets):
        
        l1_loss = self.l1_loss(inputs, targets)
        gradient_loss = self.gradient_loss(inputs, targets)
        #combining both the loss
        gradient_l1_loss = gradient_loss + l1_loss
        return gradient_l1_loss

    

class GradientDiceL1Loss(nn.Module):
    def __init__(self):
        super(GradientDiceL1Loss, self).__init__()
        self.L1_loss = nn.SmoothL1Loss(reduction='mean')
        self.gradient_dice_loss = DiceGradientLoss()
        self.sigmoid =nn.Sigmoid()
        
    def forward(self, inputs, targets, flag='Height'):

        l1_loss1 = self.L1_loss(inputs, targets)
        if(flag=='Height'):
            gradient_dice_loss = self.gradient_dice_loss(self.sigmoid(30*inputs), self.sigmoid(30*targets))
        else:
            gradient_dice_loss = self.gradient_dice_loss(inputs, targets)
        #combining both the loss
        Gradient_Dice_L1 = l1_loss1 + gradient_dice_loss
        return Gradient_Dice_L1
    
    
lossDictionary = {'Dice': DiceLoss, 'DiceRMSE': DiceRMSELoss,  'DiceL1': DiceL1Loss, 'DiceBCE': DiceBCELoss, 'Boundary': BoundaryLoss, 'BhattacharyyaDistance': BhattacharyyaDistance, 'DiceBhattacharyyaDistanceL1': DiceBhattacharyyaDistanceL1Loss, 'L1BhattacharyyaDistance': L1BhattacharyyaDistanceLoss, 'DiceBhattacharyyaDistance': DiceBhattacharyyaDistanceLoss, 'Gradient': GradientMatchingLoss, 'DiceGradient': DiceGradientLoss, 'GradientL1': GradientL1Loss, 'SmoothL1': nn.SmoothL1Loss, 'MSE': nn.MSELoss, 'GradientDiceL1': GradientDiceL1Loss}