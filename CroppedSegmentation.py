import os
import pickle
import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.utils.data as data
# from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.ops import sigmoid_focal_loss
import segmentation_models_pytorch as smp
from torchvision.ops import masks_to_boxes
from torchmetrics.classification import BinaryJaccardIndex

from torchgeo.models import ResNet18_Weights,ResNet50_Weights

#our files
import utils
from dataloader import CroppedSegmentationDataset, CroppedSegmentationPerTimeDataset
import models
# writer=SummaryWriter(log_dir='logs/exp3', max_queue=2)


#Parameters
args = utils.parse_arguments(True)

if args.pretrained:
    if args.in_channels == 3:
        weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
    else:
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
else:
    weights = None
    
if args.resume_training:
    checkpoint_path = args.checkpoint  # Replace with your file path
else:
    checkpoint_path = None

    
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
utils.set_seed(seed)


#DATALOADER
geo_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
#     transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
])

if in_channels==3:
    color_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])
else:
    color_transform=None

#not using resizing at the moment. But can be passed in the Dataset Class Object    
image_resize = transforms.Compose([transforms.Resize(args.upsampled_image_size,transforms.InterpolationMode.BICUBIC, antialias=True)])
mask_resize = transforms.Compose([transforms.Resize(args.upsampled_mask_size,transforms.InterpolationMode.NEAREST, antialias=True)])


mapping_df = pd.read_csv('dataset/MappingCroppedPerTimeToFullImage.csv')
train_set = list(mapping_df.loc[mapping_df.Set =='Train', 'CroppedPerTime'])
val_set = list(mapping_df.loc[mapping_df.Set =='Val', 'CroppedPerTime'])
print(len(train_set), len(val_set))


train_dataset = CroppedSegmentationPerTimeDataset(data_dir = args.data_dir, image_files=train_set, in_channels=args.in_channels, geo_transforms=geo_transform, color_transforms= color_transform, use_timepoints=args.use_timepoints, normalizing_factor=args.normalizing_factor)
val_dataset = CroppedSegmentationPerTimeDataset(data_dir = args.data_dir, image_files=val_set, in_channels=args.in_channels, geo_transforms=None, color_transforms= None, use_timepoints=args.use_timepoints, normalizing_factor=args.normalizing_factor)

train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory = True, num_workers = args.workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  pin_memory = True,  num_workers = args.workers)


#MODELS
model = models.setup_model(kind=args.model_type, in_channels=args.in_channels, pretrained=args.pretrained, pretrained_weights=weights, resume=args.resume_training, checkpoint_path=checkpoint_path)

#FREEZE MODEL
# for name, param in model.named_parameters():
#     if name.split('.')[0]=='encoder':
#         param.requires_grad=False


if args.loss == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif args.loss == 'DICE':
    criterion = smp.losses.DiceLoss(mode ='binary')
iou_metric = BinaryJaccardIndex(0.5)      
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 15, threshold=0.01, threshold_mode='rel', cooldown=2, min_lr=1e-7, eps=1e-08, verbose=True)


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
    model = model.to(device)
    iou_metric = iou_metric.to(device)
else:
    device = torch.device("cpu")


training_losses = []
validation_losses = []
validation_ious = []
min_val_loss = np.inf
counter = 0

for e in range(starting_epoch,starting_epoch+N_EPOCHS):    
    #Training
    print("Starting Epoch : ",str(e))
    train_loss=0
    model.train();
    for i , batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        data, target = batch[0].to(device).float(), batch[1].to(device)
        output = model(data)        
        if loss_type == 'FOCAL':
            loss = sigmoid_focal_loss(output, target, alpha = 0.75, reduction = 'mean')
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i%500==0:
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
                loss = sigmoid_focal_loss(output, target, alpha = 0.75, reduction = 'mean')
            else:
                loss = criterion(output, target)
            output = torch.sigmoid(output)
#             pred_mask = (output > 0.5).float()
            iou = iou_metric(output, target)
            val_iou+=iou.item()
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
    
    if (e+1)%15==0:
        print("Training Losses")
        print(training_losses)
        print("Val Losses")
        print(validation_losses)
        print("Val IOUs")
        print(validation_ious)
        
torch.save(model.state_dict(), os.path.join(model_dir, "end.pth"))

print("Training Losses")
print(training_losses)
print("Val Losses")
print(validation_losses)
print("Val IOUs")
print(validation_ious)



