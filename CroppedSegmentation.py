#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import random 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.ops import sigmoid_focal_loss
import segmentation_models_pytorch as smp
from torchvision.ops import masks_to_boxes

from torchgeo.models import ResNet18_Weights,ResNet50_Weights

#our files
import utils
from dataloader import CroppedSegmentationDataset
import models


# In[2]:


# args = utils.parse_arguments()
#Parameters
data_dir= r"/scratch/kj1447/gracelab/dataset"
model_dir = os.path.join("/scratch/kj1447/gracelab/models","cropped_rgb_resnet18_augmentation_dice")
loss_type = "DICE"
N_EPOCHS = 250
BATCH_SIZE = 8
num_workers = 8
train_ratio = 0.8
learning_rate = 1e-3
lookback = 10
seed = 42
starting_epoch = 0
resume = False


# In[3]:


if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
utils.set_seed(seed)


# In[4]:


image_dir = os.path.join(data_dir, 'image_stack_cropped')
mask_dir = os.path.join(data_dir, 'mask_cropped')
image_filenames = os.listdir(image_dir)
random.shuffle(image_filenames)
train_set = image_filenames[:int(train_ratio*len(image_filenames))]
val_set = image_filenames[int(train_ratio*len(image_filenames)):]


# In[5]:


geo_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
#     transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
])

color_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

train_dataset = CroppedSegmentationDataset(data_dir = data_dir, image_files=train_set, geo_transforms=geo_transform, color_transforms= color_transform)
val_dataset = CroppedSegmentationDataset(data_dir = data_dir, image_files=val_set, geo_transforms=None, color_transforms= None)

train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True, num_workers = num_workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  pin_memory = True,  num_workers = num_workers)


# In[7]:


weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
kind = 'vanila_unet'
pretrained=True
checkpoint_path = "/scratch/kj1447/gracelab/models/cropped_rgb_resnet18_augmentation/36.pth"  # Replace with your file path
model = models.setup_model(kind=kind, pretrained=pretrained, pretrained_weights=weights, resume=resume, checkpoint_path)

for name, param in model.named_parameters():
#     if name.split('.')[0]=='encoder':
#         param.requires_grad=False
    print(name, param.requires_grad)
# In[77]:


if loss_type == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif loss_type == 'DICE':
    criterion = smp.losses.DiceLoss(mode ='binary')
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 8, threshold=0.01, threshold_mode='rel', cooldown=2, min_lr=1e-7, eps=1e-08, verbose=True)


# In[78]:


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
    model = model.to(device)
else:
    device = torch.device("cpu")


# In[79]:


import time
start = time.time()
for batch in train_dataloader:
#     print(batch[0].shape, batch[1].shape)
    break
end = time.time()
print(end-start)
print(len(train_dataloader))


# In[80]:


training_losses = []
validation_losses = []
validation_ious = []
min_val_loss = np.inf
counter = 0

for e in range(starting_epoch,starting_epoch+N_EPOCHS):    
    #Training
    train_loss=0
    model.train();
    for i , batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        data, target = batch[0].to(device).float(), batch[1].to(device)
        output = model(data)        
        if loss_type == 'FOCAL':
            loss = sigmoid_focal_loss(output, target, alpha = 0.25, reduction = 'mean')
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i%400==0:
            print("Batch ",i," - Train Loss = ",train_loss/(i+1))
    train_loss = train_loss/len(train_dataloader)
    training_losses.append(train_loss)
    
    #Validation
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        for batch in val_dataloader:
            data, target = batch[0].to(device).float(), batch[1].to(device)
            output = model(data)
            if loss_type == 'FOCAL':
                loss = sigmoid_focal_loss(output, target, alpha = 0.25, reduction = 'mean')
            else:
                loss = criterion(output, target)
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float()
            val_batch_iou = []
            for pred, true in zip(pred_mask, target):
                pred = torch.squeeze(pred, dim=0)
                true = torch.squeeze(true, dim=0)

                intersection = torch.logical_and(pred, true).sum().item()
                union = torch.logical_or(pred, true).sum().item()
                iou = intersection / union if union > 0 else 0.0
                val_batch_iou.append(iou)
            val_batch_iou = torch.tensor(val_batch_iou).mean()

            val_iou +=val_batch_iou.item()
            val_loss += loss.item()            
        val_iou = val_iou/len(val_dataloader)
        val_loss = val_loss/len(val_dataloader)
        
        if e == starting_epoch:
            min_val_loss=val_loss
            
        validation_losses.append(val_loss)
        validation_ious.append(val_iou)
        
    print("Epoch-",e,"| Training Loss - ",train_loss,", Validation Loss - ",val_loss,", Validation IOU - ", val_iou)
    
    if (e%10==0) or (val_loss < min_val_loss):
        torch.save(model.state_dict(), os.path.join(model_dir, f"{e+1}.pth"))
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter=0
    else:
        counter+=1

    #checking if LR needs to be reduced
    scheduler.step(val_loss)
    
    if counter>=lookback:
        print("Early Stopping Reached")
        break
        
torch.save(model.state_dict(), os.path.join(model_dir, "end.pth"))

print("Training Losses")
print(training_losses)
print("Val Losses")
print(validation_losses)
print("Val IOUs")
print(validation_ious)


# In[ ]:




