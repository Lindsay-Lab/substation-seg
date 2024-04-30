import os
import pickle
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
from torchmetrics.classification import BinaryJaccardIndex

from torchgeo.models import ResNet18_Weights,ResNet50_Weights

import utils
from dataloader import CroppedSegmentationDataset, CroppedSegmentationPerTimeDataset, FullImageDataset
import models

#Parameters
args = utils.parse_arguments()
data_dir= r"/scratch/kj1447/gracelab/dataset"
model_dir = os.path.join("/scratch/kj1447/gracelab/models",args.model_dir)
loss_type = args.loss
N_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
num_workers = args.workers
train_ratio = args.train_ratio
learning_rate = args.learning_rate
lookback = args.lookback
seed = args.seed
starting_epoch = args.starting_epoch
resume = args.resume_training
in_channels = args.in_channels
use_timepoints = args.use_timepoints

kind = args.model_type
pretrained=args.pretrained
checkpoint_path = args.checkpoint
# 'models/cropped_sentinel_resnet18_augmentation_focal_75/end.pth'  # Replace with your file path

if pretrained:
    if in_channels == 3:
        weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
    else:
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
else:
    weights = None


if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
utils.set_seed(seed)


image_dir = os.path.join(data_dir, 'image_stack')
mask_dir = os.path.join(data_dir, 'mask')
image_filenames = os.listdir(image_dir)
# with open("dataset/exclude_list","rb") as f:
#     exclude_list = pickle.load(f)
# image_filenames_sub = [x for x in image_filenames if x not in exclude_list]

# print(len(image_filenames), len(exclude_list), len(image_filenames_sub))
random.shuffle(image_filenames)
train_set = image_filenames[:int(train_ratio*len(image_filenames))]
val_set = image_filenames[int(train_ratio*len(image_filenames)):]
print(len(train_set), len(val_set))


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

train_dataset = FullImageDataset(data_dir = data_dir, image_files=train_set, in_channels=in_channels, geo_transforms=geo_transform, color_transforms= color_transform, use_timepoints=use_timepoints)
val_dataset = FullImageDataset(data_dir = data_dir, image_files=val_set, in_channels=in_channels, geo_transforms=None, color_transforms= None, use_timepoints=use_timepoints)

train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True, num_workers = num_workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  pin_memory = True,  num_workers = num_workers)



model = models.setup_model(kind=kind, in_channels=in_channels, pretrained=pretrained, pretrained_weights=weights, resume=resume, checkpoint_path=checkpoint_path)

if loss_type == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif loss_type == 'DICE':
    criterion = smp.losses.DiceLoss(mode ='binary')

iou_metric = BinaryJaccardIndex(0.5)    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 8, threshold=0.01, threshold_mode='rel', cooldown=2, min_lr=1e-7, eps=1e-08, verbose=True)



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
    train_loss=0
    model.train();
    for i , batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        data, target = batch[0].to(device).float(), batch[1].to(device)
        output = model(data[:,:,2:-2,2:-2])        
        if loss_type == 'FOCAL':
            loss = sigmoid_focal_loss(output, target[:,:,2:-2,2:-2], alpha = 0.75, reduction = 'mean')
        else:
            loss = criterion(output, target[:,:,2:-2,2:-2])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i%100==0:
            print("Batch ",i," - Train Loss = ",train_loss/(i+1))
    train_loss = train_loss/len(train_dataloader)
    training_losses.append(train_loss)
    
#     writer.add_scalar("Loss/Train", train_loss, e)
    
    #Validation
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        for batch in val_dataloader:
            data, target = batch[0].to(device).float(), batch[1].to(device)
            output = model(data[:,:,2:-2,2:-2])
            if loss_type == 'FOCAL':
                loss = sigmoid_focal_loss(output, target[:,:,2:-2,2:-2], alpha = 0.75, reduction = 'mean')
            else:
                loss = criterion(output, target[:,:,2:-2,2:-2])
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float()
            iou = iou_metric(pred_mask, target[:,:,2:-2,2:-2])
            val_iou+=iou.item()
            
            val_loss += loss.item()            
        val_iou = val_iou/len(val_dataloader)
        val_loss = val_loss/len(val_dataloader)
        
        if e == starting_epoch:
            min_val_loss=val_loss
            
        validation_losses.append(val_loss)
        validation_ious.append(val_iou)
#         writer.add_scalar("Loss/Validation", val_loss, e)
#         writer.add_scalar("IoU/Validation", val_iou, e)
        
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
    
    if (e+1)%10==0:
        print("Training Losses")
        print(training_losses)
        print("Val Losses")
        print(validation_losses)
        print("Val IOUs")
        print(validation_ious)
        
torch.save(model.state_dict(), os.path.join(model_dir, "end.pth"))
# writer.flush()

print("Training Losses")
print(training_losses)
print("Val Losses")
print(validation_losses)
print("Val IOUs")
print(validation_ious)





