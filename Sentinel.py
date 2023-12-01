import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.ops import sigmoid_focal_loss
import segmentation_models_pytorch as smp
from torchgeo.models import ResNet50_Weights,ResNet50_Weights

#our files
import utils
from dataloader import FullImageDataset
import models


#Parameters
args = utils.parse_arguments()
data_dir= r"/scratch/paa9751/substation-seg/dataset"
model_dir = os.path.join("/scratch/paa9751/substation-seg/models/",args.model_dir)
loss_type = args.loss
N_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
num_workers = args.workers
train_ratio = args.train_ratio
learning_rate = args.learning_rate
lookback = args.lookback
seed = args.seed
resume = args.resume_training
starting_epoch = args.starting_epoch
kind = args.model_type
upsampled_image_size=args.upsampled_image_size
upsampled_mask_size=args.upsampled_mask_size
in_channels = args.in_channels

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
utils.set_seed(seed)


#DATALOADER
image_transform = transforms.Compose([
    transforms.Resize(upsampled_image_size,transforms.InterpolationMode.BICUBIC, antialias=True)
])

mask_transform = transforms.Compose([
    transforms.Resize(upsampled_mask_size,transforms.InterpolationMode.NEAREST, antialias=True)
])

downsample = transforms.Resize(upsampled_mask_size,transforms.InterpolationMode.BILINEAR, antialias=True) # for downsampling output mask in train loop

image_dir = os.path.join(data_dir, 'image_stack')
mask_dir = os.path.join(data_dir, 'mask')
image_filenames = os.listdir(image_dir)
random.shuffle(image_filenames)
train_set = image_filenames[:int(train_ratio*len(image_filenames))]
val_set = image_filenames[int(train_ratio*len(image_filenames)):]

train_dataset = FullImageDataset(data_dir = data_dir, image_files=train_set, in_channels = in_channels, img_transforms=image_transform, mask_transforms=mask_transform)
val_dataset = FullImageDataset(data_dir = data_dir, image_files=val_set, in_channels = in_channels, img_transforms=image_transform, mask_transforms=mask_transform)

train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True, num_workers = num_workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  pin_memory = True,  num_workers = num_workers)


#MODEL
pretrained=True
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
checkpoint_path = None
model = setup_model(kind, pretrained=pretrained, pretrained_weights=weights, resume=resume, checkpoint_path=checkpoint_path)

#FREEZE MODEL
# for name, param in model.named_parameters():
#     if name.split('.')[0]=='encoder':
#         param.requires_grad=False

if loss_type == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif loss_type == 'DICE':
    criterion = smp.losses.DiceLoss(mode ='binary')
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 15, threshold=0.01, threshold_mode='rel', cooldown=10, min_lr=1e-7, eps=1e-08, verbose=True)


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
    model = model.to(device)
else:
    device = torch.device("cpu")

#TRAINING & VALIDATION
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
        if (upsampled_mask_size<upsampled_image_size) and (kind =='vanilla_unet'):
            output = downsample(output) 
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
            if (upsampled_mask_size<upsampled_image_size) and (kind =='vanilla_unet'):
                output = downsample(output) 
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
