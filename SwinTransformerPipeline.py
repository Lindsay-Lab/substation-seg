import os
import random
import numpy as np
import pickle
import wandb

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryJaccardIndex
# import satlaspretrain_models

#our files
import utils
from dataloader import FullImageDataset

from models import setup_model


args = utils.parse_arguments(True)
wandb.init(
    # set the wandb project where this run will be logged
    project= args.exp_name,
    name = "run_"+str(args.exp_number),
    # track hyperparameters and run metadata
    config=args
)

if not os.path.isdir(args.model_dir):
    os.mkdir(args.model_dir)
# utils.set_seed(seed)

torch.manual_seed(args.seed)
random.seed(args.seed)


#DATALOADER
geo_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
#     transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
])
color_transform=None
# if args.in_channels==3:
#     color_transform = transforms.Compose([
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     ])
# else:
#     color_transform=None

image_resize = transforms.Compose([transforms.Resize(args.upsampled_image_size,transforms.InterpolationMode.BICUBIC, antialias=True)])
mask_resize = transforms.Compose([transforms.Resize(args.upsampled_mask_size,transforms.InterpolationMode.NEAREST, antialias=True)])

# image_resize = None
# mask_resize = None

image_dir = os.path.join(args.data_dir, 'image_stack')
mask_dir = os.path.join(args.data_dir, 'mask')

# for multi-image
if args.use_timepoints:
    with open("dataset/four_or_more_timepoints.pkl",'rb') as f:
        image_filenames = pickle.load(f)
else:
    image_filenames = os.listdir(image_dir)

random.Random(args.seed).shuffle(image_filenames)

train_set = image_filenames[:int(args.train_ratio*len(image_filenames))]
val_set = image_filenames[int(args.train_ratio*len(image_filenames)):]

# data_dir, image_files, in_channels=3, geo_transforms=None, color_transforms= None, use_timepoints=False, normalizing_factor = 5000
train_dataset = FullImageDataset(data_dir = args.data_dir, image_files=train_set, in_channels = args.in_channels,\
                                 normalizing_factor=args.normalizing_factor, geo_transforms=geo_transform, \
                                 color_transforms= color_transform, image_resize=image_resize, mask_resize=mask_resize,\
                                 mask_2d=True, use_timepoints=args.use_timepoints, model_type = args.model_type)
val_dataset = FullImageDataset(data_dir = args.data_dir, image_files=val_set, in_channels = args.in_channels,\
                               normalizing_factor=args.normalizing_factor, geo_transforms=None, color_transforms= None,\
                               image_resize=image_resize, mask_resize=mask_resize, mask_2d=True, use_timepoints=args.use_timepoints, model_type = args.model_type)

train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory = True, num_workers = args.workers, drop_last=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  pin_memory = True,  num_workers = args.workers, drop_last=True)


#MODEL
model = setup_model(args)


optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 5, threshold=0.01, threshold_mode='rel', cooldown=10, min_lr=1e-7, eps=1e-08)
iou_metric = BinaryJaccardIndex(0.5) 


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
    model = model.to(device)
    iou_metric = iou_metric.to(device)
else:
    device = torch.device("cpu")



#TRAINING & VALIDATION
training_losses = []
validation_losses = []
validation_ious = []
learning_rates=[]
min_val_loss = np.inf
counter = 0

for e in range(args.starting_epoch,args.starting_epoch+args.epochs):    
    #Training Loop
    train_loss=0
    model.train()
    
    for i , batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        data, target = batch[0].to(device).float(), batch[1].to(device)
        output,loss = model(data, target)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i%100==0:
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
            output,loss = model(data, target)
#             output = torch.sigmoid(output)
#             pred_mask = (output > 0.5).float()
            iou = iou_metric(torch.argmax(output, dim=1), torch.argmax(target, dim = 1))
            val_iou+=iou.item()
            val_loss += loss.item()   
        val_iou = val_iou/len(val_dataloader)
        val_loss = val_loss/len(val_dataloader)
        
        if e == args.starting_epoch:
            min_val_loss=val_loss
            
        validation_losses.append(val_loss)
        validation_ious.append(val_iou)
        
    print("Epoch-",e,"| Training Loss - ",train_loss,", Validation Loss - ",val_loss,", Validation IOU - ", val_iou)
    wandb.log({"Training Loss": train_loss, "Validation Loss": val_loss, "Validation IoU":val_iou})
    
    if (e%10==0) or (val_loss < min_val_loss):
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"{e+1}.pth"))
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter=0
    else:
        counter+=1

    #checking if LR needs to be reduced
    scheduler.step(val_loss)
    learning_rates.append(scheduler.get_last_lr()[0])
    
    if counter>=args.lookback:
        print("Early Stopping Reached")
        break
        
torch.save(model.state_dict(), os.path.join(args.model_dir, "end.pth"))

print("train_loss = ",training_losses)
print("val_loss = ",validation_losses)
print("val_iou = ", validation_ious)
print("learning_rates = ", learning_rates)


wandb.finish()


