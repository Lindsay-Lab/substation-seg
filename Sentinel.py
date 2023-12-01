import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import jaccard_score
from torchgeo.models import FarSeg
import torchgeo
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.ops import sigmoid_focal_loss
import segmentation_models_pytorch as smp

from torchgeo.models import ResNet50_Weights
from torchvision.models._api import WeightsEnum
import torch.nn.functional as F

#setting seed for reproducibility
torch.manual_seed(42)
        
parser = argparse.ArgumentParser(description = "Define parameters like loss function, model_directiory etc")

parser.add_argument("-m","--model_dir", help="Model Directory for Saving Checkpoints. A folder with this name would be created within the models folder")
parser.add_argument("-l","--loss", default = "BCE", help="Can take following values - BCE, FOCAL or DICE")
parser.add_argument("-e","--epochs", default = 250, type = int, help="Number of epochs")
parser.add_argument("-bs","--batch_size", default = 32, type=int, help="Batch Size")
parser.add_argument("-w","--workers", default = 16, type=int, help="Number of Workers")
parser.add_argument("-tr", "--train_ratio", default = 0.8, type = float, help = "Train-Validation Ratio")
parser.add_argument("-lr", "--learning_rate", default = 1e-3, type = float, help = "Learning Rate")

parser.add_argument("-model_type", default='vanilla_unet', help='type of model')

args = parser.parse_args()

print(args)

#Parameters
data_dir= r"/scratch/kj1447/gracelab/dataset"
model_dir = os.path.join("/scratch/kj1447/gracelab/models",args.model_dir)
loss_type = args.loss
N_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
num_workers = args.workers
train_ratio = args.train_ratio
learning_rate = args.learning_rate

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

image_transform = transforms.Compose([
    transforms.Resize((256,256),transforms.InterpolationMode.BICUBIC, antialias=True)
])

mask_transform = transforms.Compose([
    transforms.Resize((256,256),transforms.InterpolationMode.NEAREST, antialias=True)
])

class CustomDataset(data.Dataset):
    def __init__(self, data_dir, img_transforms=None, mask_transforms=None):
        self.data_dir = data_dir
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.image_dir = os.path.join(data_dir, 'image_stack')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.image_filenames = os.listdir(self.image_dir)
        
    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = np.median(np.load(image_path)['arr_0'], axis = 0) 
#         image = image[[3,2,1],:,:]

        #standardizing image
        image = image/5000
        
        mask = np.load(mask_path)['arr_0']
        mask[mask != 3] = 0
        mask[mask == 3] = 1
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).float()
        mask = mask.unsqueeze(dim=0)
        
        if self.img_transforms:
            image = self.img_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
            
        return image, mask

    def __len__(self):
        return len(self.image_filenames)

    def plot(self):
        ##Needs to be updated
        index = np.random.randint(0,self.__len__())
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = np.median(np.load(image_path)['arr_0'], axis =0)
        mask = np.load(mask_path)['arr_0']
        
        print(np.unique(mask, return_counts = True))
        
        image = image[[3, 2, 1], :, :]
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image/5000, 0,1)

        fig, axs = plt.subplots(1,2)
        axs[0].imshow(image)
        axs[1].imshow(mask,)

        

#Loading the dataloaders
dataset = CustomDataset(data_dir=data_dir, img_transforms=image_transform, mask_transforms=mask_transform)
val_ratio = 1 - train_ratio

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True, num_workers = num_workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers = num_workers)


weights = ResNet50_Weights.SENTINEL2_ALL_MOCO

model = setup_model(kind, pretrained=True, pretrained_weights=weights, resume=False, checkpoint_path=None)

###########################
#code to itialize weights
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)

# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
# net.apply(init_weights)
################################

# checkpoint_path = "/scratch/kj1447/gracelab/models/Sentinel_Scheduler/126.pth"  # Replace with your file path
# checkpoint = torch.load(checkpoint_path)

# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in checkpoint.items():
#     name = k[7:] # remove `module.module.`
#     new_state_dict[name] = v
# # load params
# model.load_state_dict(checkpoint)

#if isinstance(weights, WeightsEnum):
    #state_dict = weights.get_state_dict(progress=True)
#model.encoder.load_state_dict(state_dict)

if loss_type == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif loss_type == 'DICE':
    criterion = smp.losses.DiceLoss(mode ='binary')
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.2, patience = 15, threshold=0.01, threshold_mode='rel', cooldown=10, min_lr=1e-7, eps=1e-08, verbose=True)


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
    #multiple gpu training
#     model = nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)
#     criterion = criterion.to(device)
else:
    device = torch.device("cpu")

training_losses = []
validation_losses = []
validation_ious = []
min_val_loss = 0

starting_epoch = 0
for e in range(starting_epoch,starting_epoch+N_EPOCHS):    
    #Training
    train_loss=0
    model.train();
    for batch in train_dataloader:
        optimizer.zero_grad()
        data, target = batch[0].to(device).float(), batch[1].to(device)
        output = model(data)
        
        if loss_type == 'FOCAL':
            loss = sigmoid_focal_loss(output, target, alpha = 0.99, reduction = 'mean')
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
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
                loss = sigmoid_focal_loss(output, target, alpha = 0.99, reduction = 'mean')
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
        
        if e ==0:
            min_val_loss=val_loss
            
        validation_losses.append(val_loss)
        validation_ious.append(val_iou)
        
    print("Epoch-",e,"| Training Loss - ",train_loss,", Validation Loss - ",val_loss,", Validation IOU - ", val_iou)
    
    if (e%10==0) or (val_loss < min_val_loss):
        torch.save(model.state_dict(), os.path.join(model_dir, f"{e+1}.pth"))
    
    if val_loss < min_val_loss:
        min_val_loss = val_loss
    
    #checking if LR needs to be reduced
    scheduler.step(val_loss)
    
torch.save(model.state_dict(), os.path.join(model_dir, "end.pth"))

print("Training Losses")
print(training_losses)
print("Val Losses")
print(validation_losses)
print("Val IOUs")
print(validation_ious)
