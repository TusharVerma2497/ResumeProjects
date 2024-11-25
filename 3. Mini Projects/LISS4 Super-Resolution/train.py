from dataset import DelhiDataset
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch
from models import Generator, Discriminator
from loss import GeneratorLoss, TVLoss, DiceLoss
from tqdm import tqdm
from math import log10


" CREATING TRAINING DATASET "

# sentinel2RasterPath = "/home/nximish/footprint_extraction/ExperimentsAndMetaData/Experiments/S2CompositeDelhi2024TypeS2_SR_HARMONIZED.tif"
google_rootPath = '/home/nximish/footprint_extraction'
LISS4_rootPath = '/home/nximish/LISS4Upsampling/Dataset'

dataset = DelhiDataset(google_rootPath, LISS4_rootPath)
batchSize = 32

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Dataset size and indices
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

# Calculate the split sizes
train_split = int(np.floor(0.8  * dataset_size))
val_split = int(np.floor(0.1  * dataset_size))

# Create indices for each split
train_indices = indices[:train_split]
val_indices = indices[train_split:train_split + val_split]

# Create samplers for each subset
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create loaders for training and validation
train_loader = DataLoader(dataset, batch_size=batchSize, sampler=train_sampler, shuffle=False, num_workers=12, pin_memory=True)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler, shuffle=False, pin_memory=True, num_workers=5)

print('height dataset train:val split')
print(len(train_loader)*batchSize, len(val_loader))




" CREATING TRAINING MODEL "

loadModel = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(loadModel):
    #loading generator
    generator_1 = Generator(4, maskOutput=True)
    state_dict = torch.load('saved_state/netG1.pth')
    generator_1.load_state_dict(state_dict)
    #loading discriminator
    discriminator_1 = Discriminator()
    state_dict = torch.load('saved_state/netD1.pth')
    discriminator_1.load_state_dict(state_dict)
    
    #sending to devive
    generator_1 = generator_1.to(device)
    discriminator_1 = discriminator_1.to(device)
    
    #loading Generator optimizer
    optimizer_g1 = torch.optim.Adam(generator_1.parameters(), lr=1e-4, betas=(0.9, 0.999))
    state_dict = torch.load('saved_state/optG1.pth')
    optimizer_g1.load_state_dict(state_dict)
    #loading Discriminator optimizers
    optimizer_d1 = torch.optim.Adam(discriminator_1.parameters(), lr=1e-4, betas=(0.9, 0.999))
    state_dict = torch.load('saved_state/optD1.pth')
    optimizer_d1.load_state_dict(state_dict)
    print('models loaded')
    
else:
    # Creating generator (4x) and discriminator
    generator_1 = Generator(4, maskOutput=True).to(device)
    discriminator_1 = Discriminator().to(device)

    # Defining the optimizers
    optimizer_g1 = torch.optim.Adam(generator_1.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_d1 = torch.optim.Adam(discriminator_1.parameters(), lr=1e-4, betas=(0.9, 0.999))
    print('models Initialized')

# Generator criteria
generator_criterion = GeneratorLoss().to(device)
dloss = DiceLoss().to(device)




" Training Function "

def adverserialTrainOneEpoch(train_loader):
    gen1Loss=0
    dis1Loss=0
    epsilon = 1e-8
    
    for index, (low_r, high_r, mask_GT) in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_size = low_r.size(0)
        low_r = low_r.float().to(device)
        high_r = high_r.float().to(device)
        mask_GT = mask_GT.float().to(device)
        
#       ############################
#       # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
#       ###########################
 
        #generating fake min_r (passing through first generator)
        fake_high_r, latent_g1, mask = generator_1(low_r, return_dense = True)
        #passing through first discriminator for fake labels
        fake_high_r_labels = discriminator_1(fake_high_r).mean()
        #calculating loss and backpropagation
        optimizer_g1.zero_grad()
        g1_loss = generator_criterion(fake_high_r_labels, fake_high_r, high_r) + 0.006*dloss(mask, mask_GT)
        g1_loss.backward()
        optimizer_g1.step()
        
        
#       ############################
#       # (2) Update D network: maximize D(x)-1-D(G(z))
#       ###########################

        #passing through first discriminator for discriminator loss
        real_high_r_labels = discriminator_1(high_r).mean()
        fake_high_r_labels = discriminator_1(fake_high_r.detach()).mean()
#         d1_loss = 1 - real_mid_r_labels + fake_mid_r_labels
        d1_loss = -(torch.log(real_high_r_labels + epsilon) + torch.log(1 - fake_high_r_labels + epsilon))
        #calculating loss and backpropagation
        optimizer_d1.zero_grad()
        d1_loss.backward()
        optimizer_d1.step()

        gen1Loss+=g1_loss.item()
        dis1Loss+=d1_loss.item()
        
        if(index%100 == 0):
            print(f'Gen1Loss: {gen1Loss}, Dis1Loss: {dis1Loss}')
            gen1Loss=0
            dis1Loss=0
    

    
    
" Validation Function "
    
def validation(val_loader):
    with torch.no_grad():
        gen1_mse_total = 0
        num_samples = 0

        for val_lr, val_hr, _ in tqdm(val_loader):
            batch_size = val_lr.size(0)
            lr = val_lr.float().to(device)
            hr = val_hr.float().to(device)
            
            # generating fake mid_r images with first generator
            fake_hr, latent_g1, _ = generator_1(lr, return_dense = True)
            # Compute MSE for the first generator
            batch_mse = ((fake_hr - hr) ** 2).data.mean()
            gen1_mse_total += batch_mse * batch_size

            num_samples += batch_size

        # Avg MSE
        gen1_mse_total = gen1_mse_total / num_samples
        # Compute PSNR using the maximum pixel value
        max_pixel = 1.0  # If images are normalized to [0, 1], otherwise use 255 for [0, 255]
        gen1_psnr = 10 * log10((max_pixel**2) / gen1_mse_total)

        return gen1_mse_total.item(), gen1_psnr



    
" Training Loop "

NUM_EPOCHS = 50
    
for epoch in range(1, NUM_EPOCHS + 1):
    print(f'epoch : {epoch} started')
    generator_1.train()
    discriminator_1.train()
    
    #running one epoch
    adverserialTrainOneEpoch(train_loader)
    
    #saving the model
    torch.save(generator_1.state_dict(), 'saved_state/netG1.pth')
    torch.save(discriminator_1.state_dict(), 'saved_state/netD1.pth')
    #saving the optimizer
    torch.save(optimizer_g1.state_dict(), 'saved_state/optG1.pth')
    torch.save(optimizer_d1.state_dict(), 'saved_state/optD1.pth')
    
    
    #running validation
    print('validation')
    generator_1.eval()
    discriminator_1.eval()
    gen1_mse, gen1_psnr = validation(val_loader)
    
    print(f'Gen1_MSE: {gen1_mse}, Gen1_psnr: {gen1_psnr}')
    
    
