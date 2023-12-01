import torch
import numpy as np
import matplotlib.pyplot as plt
import os


class CroppedSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_files, in_channels=3, geo_transforms=None, color_transforms= None, use_timepoints=False):
        self.data_dir = data_dir
        self.geo_transforms = geo_transforms
        self.color_transforms = color_transforms
        self.image_dir = os.path.join(data_dir, 'image_stack_cropped')
        self.mask_dir = os.path.join(data_dir, 'mask_cropped')
        self.image_filenames = image_files
        self.in_channels=in_channels
        self.use_timepoints = use_timepoints
        
    def __getitem__(self, index):
        # load images and masks
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        if self.use_timepoints: 
            image = np.load(image_path) # t x 13 x h x w 
            image = np.reshape(image, (-1, image.shape[2], image.shape[3]))
        else: 
            image = np.median(np.load(image_path), axis=0)
            
        if self.in_channels==3:
            image = image[[3,2,1],:,:]
        
        image = np.clip(image/5000,0,1)
        
        mask = np.load(mask_path)
        image = torch.from_numpy(image) #3x228x228
        mask = torch.from_numpy(mask).float().unsqueeze(0) #1x228x228
        
        if self.geo_transforms:
            combined = torch.cat((image,mask), 0)
            combined = self.geo_transforms(combined)
            image,mask = torch.split(combined, [self.in_channels,1], 0)
        
        if self.color_transforms:
            image = self.color_transforms(image)
        
        
        return image, mask

    def __len__(self):
        return len(self.image_filenames)

    
    def plot(self):
        index = np.random.randint(0, self.__len__())
        image, mask = self.__getitem__(index)
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(image.permute(1,2,0))
        axs[1].imshow(image.permute(1,2,0))
        axs[1].imshow(mask.permute(1,2,0), alpha=0.5, cmap='gray')
        

class FullImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_files, in_channels=3, img_transforms=None, mask_transforms=None, use_timepoints=False):
        self.data_dir = data_dir
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.image_dir = os.path.join(data_dir, 'image_stack')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.image_filenames = image_files
        self.in_channels=in_channels
        self.use_timepoints = use_timepoints 
        
    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename
        mask_path = os.path.join(self.mask_dir, mask_filename)

        if self.use_timepoints: 
            image = np.load(image_path)['arr_0'] # t x 13 x h x w 
            image = np.reshape(image, (-1, image.shape[2], image.shape[3]))
        else: 
            image = np.median(np.load(image_path)['arr_0'], axis=0)
            
        if self.in_channels==3:
            image = image[[3,2,1],:,:]

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
        index = np.random.randint(0, self.__len__())
        image, mask = self.__getitem__(index)
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(image.permute(1,2,0))
        axs[1].imshow(image.permute(1,2,0))
        axs[1].imshow(mask.permute(1,2,0), alpha=0.5, cmap='gray')


