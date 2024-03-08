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
from torchmetrics.classification import BinaryJaccardIndex

#our files
import utils
from dataloader import FullImageDataset
import models
from models import setup_model

#Parameters
args = utils.parse_arguments(True)

if not os.path.isdir(args.model_dir):
    os.mkdir(args.model_dir)
utils.set_seed(args.seed)


#DATALOADER
image_resize = transforms.Compose([transforms.Resize(args.upsampled_image_size,transforms.InterpolationMode.BICUBIC, antialias=True)])
mask_resize = transforms.Compose([transforms.Resize(args.upsampled_mask_size,transforms.InterpolationMode.NEAREST, antialias=True)])


image_dir = os.path.join(args.data_dir, 'image_stack')
mask_dir = os.path.join(args.data_dir, 'mask')
image_filenames = os.listdir(image_dir)
random.Random(args.seed).shuffle(image_filenames)
train_set = image_filenames[:int(args.train_ratio*len(image_filenames))]
val_set = image_filenames[int(args.train_ratio*len(image_filenames)):]

train_dataset = FullImageDataset(data_dir = args.data_dir, image_files=train_set, in_channels = args.in_channels, normalizing_factor=args.normalizing_factor, image_resize=image_resize, mask_resize=mask_resize)
val_dataset = FullImageDataset(data_dir = args.data_dir, image_files=val_set, in_channels = args.in_channels, normalizing_factor=args.normalizing_factor, image_resize=image_resize, mask_resize=mask_resize)

train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory = True, num_workers = args.workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  pin_memory = True,  num_workers = args.workers)


#MODEL

weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
model = setup_model(args.model_type, in_channels=args.in_channels , pretrained=args.pretrained, pretrained_weights=weights, resume=args.resume_training, checkpoint_path=args.checkpoint)

#FREEZE MODEL
# for name, param in model.named_parameters():
#     if name.split('.')[0]=='encoder':
#         param.requires_grad=False

if args.loss == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif args.loss == 'DICE':
    criterion = smp.losses.DiceLoss(mode ='binary')
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 15, threshold=0.01, threshold_mode='rel', cooldown=10, min_lr=1e-7, eps=1e-08, verbose=True)
iou_metric = BinaryJaccardIndex(0.5) 
iou_metric = iou_metric.to(device)

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
    model = model.to(device)
    criterion = criterion.to(device)
else:
    device = torch.device("cpu")

#TRAINING & VALIDATION
training_losses = []
validation_losses = []
validation_ious = []
min_val_loss = np.inf
counter = 0

for e in range(args.starting_epoch,args.starting_epoch+args.epochs):    
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
            loss = sigmoid_focal_loss(output, target, alpha = 0.75, reduction = 'mean')
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
            
            iou = iou_metric(output, target)
            val_iou+=iou.item()
            val_loss += loss.item()
 
        val_iou = val_iou/len(val_dataloader)
        val_loss = val_loss/len(val_dataloader)
        
        if e == args.starting_epoch:
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

print("train_loss = ",training_losses)
print("val_loss = ",validation_losses)
print("val_iou = ", validation_ious)

