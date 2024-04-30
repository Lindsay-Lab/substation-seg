import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torchgeo
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

data_dir= r"/scratch/paa9751/grace-lab/dataset"

class CroppedSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, box_size, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.box_size = box_size
        self.image_dir = os.path.join(data_dir, 'image_stack')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.image_filenames = os.listdir(self.image_dir)
        
    def __getitem__(self, index):
        # load images and masks
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        image = np.median(np.load(image_path)['arr_0'], axis=0)
        image = image[[3,2,1],:,:]
        image = image/5000
        
#         print(image.shape)
        
        mask = np.load(mask_path)['arr_0']
        mask[mask != 3] = 0
        mask[mask == 3] = 1
        
        image = torch.from_numpy(image) #3x228x228
        mask = torch.from_numpy(mask).float().unsqueeze(0) #1x228x228
        
    
#         # instances are encoded as different colors
#         obj_ids = torch.unique(mask) #[0,1]
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:] #[1]
#         num_objs = len(obj_ids) #1 

#         # split the color-encoded mask into a set
#         # of binary masks
#         masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        
#         print("OG MASK",mask.shape)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(mask).int() # 1 x 4 
        
#         print("box returned from pytorch",boxes)
        
#         plt.imshow(draw_bounding_boxes((image * 255).type(torch.uint8),boxes).permute(1,2,0))
#         plt.show()
#         plt.imshow(mask.permute(1,2,0))
#         plt.show()
        
        x1 = boxes[0,0].item()
        y1 = boxes[0,1].item()
        x2 = boxes[0, 2].item()
        y2 = boxes[0, 3].item()
#         print(x1,y1,x2,y2)    
        centers = [int(x1 + 0.5*np.abs(x1 - x2)), int(y1 + 0.5*np.abs(y1 - y2))]
        
#         print(centers)
        
        offset = self.box_size//2
        
        x1_new = centers[0] - offset
        y1_new = centers[1] - offset
        x2_new = centers[0] + offset
        y2_new = centers[1] + offset
        print("old coords")
        print(x1_new, y1_new)
        print(x2_new, y2_new)
        #checks: 
        if x1_new<0: 
            x2_new += np.abs(x1_new) #add additional width 
            x1_new=0
        elif x2_new > mask.shape[2]: 
            x1_new = x1_new - np.abs(mask.shape[2] - x2_new) 
            x2_new = mask.shape[2] 
            
        if y1_new < 0: 
            y2_new += np.abs(y1_new)
            y1_new = 0 
        elif y2_new > mask.shape[1]:
            y1_new = y1_new - np.abs(mask.shape[1] - y2_new) 
            y2_new = mask.shape[1] 
            
#         print("new coords")
#         print(x1_new, y1_new)
#         print(x2_new, y2_new)
            
#         print("BOX AREA=", (x2_new-x1_new)*(y2_new-y1_new))
    
        
        image_cropped = image[:, y1_new:y2_new, x1_new:x2_new]
        mask_cropped = mask[:, y1_new:y2_new, x1_new:x2_new]
        
        new_boxes = torch.tensor([x1_new,y1_new, x2_new,y2_new]).unsqueeze(0)
        
#         plt.imshow(draw_bounding_boxes((image * 255).type(torch.uint8),new_boxes).permute(1,2,0))
#         plt.show()
#         plt.imshow(image_cropped.permute(1,2,0))
#         plt.show()
#         plt.imshow(mask_cropped.permute(1,2,0))
#         plt.show()
        
#         print(image_cropped.shape)
#         print(mask_cropped.shape)

        if self.transforms is not None:
            image_cropped, mask_cropped = self.transforms(image_cropped, mask_cropped)

        return image_cropped, mask_cropped

    def __len__(self):
        return len(self.image_filenames)
     
    
def get_dataloaders(BATCH_SIZE = 64, CustomDataset = CroppedSegmentationDataset):
    #Loading the dataloaders
    dataset = CustomDataset(data_dir=data_dir, transforms=None)
    train_ratio = 0.8
    val_ratio = 1 - train_ratio

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True, num_workers = 16)
    val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True, num_workers = 16)
    
    return train_dataloader, val_dataloader 

