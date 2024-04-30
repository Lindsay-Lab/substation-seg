# Substation segmentation
The aim of this project is to build a computer vision model for segmenting substations in Sentinel 2 satellite imagery. This directory contains code to train and evaluate multiple kinds of segmentation models. We experiment with different kinds of architectures, losses and training strategies. 

## The directory structure:
- dataset
  - image_stack
  - mask
- dataloader.py - creates dataloader for the training
- models.py - creates and instantiates different kinds of models
- utils.py - stores the helped functions
- Sentinel.py - training script for unet models
- SwinTransformerPipeline.py - training script for swin models 

## Dataset Details:
- dataset/image_stack and dataset/mask folders contain all the images and masks respectively.
- There are a a total of 26522 images-mask pairs stored as numpy files.
- Each image is multi-temporal and contains multiple shots taken at the same place during different revisits. Majority of the files contains images from 5 revisits.
- Each image is multi-spectral and  contains 13 channels
- Spatial Resolution of each image is 228*228.

  
Sample image and mask pair are given below -
<p align="center" width="100%">
  <img width="640px" src="https://github.com/Lindsay-Lab/substation-seg/blob/main/artifacts/example_input.png">
</p>


## Running Training Scripts
For Training UNet- 
```
python3 Sentinel.py --model_dir <directory to save models> --batch_size <batch size for dataloader> --workers <number of parallel workers> --learning_rate 1e-3 --upsampled_mask_size <size of reshaped mask> --upsampled_image_size <size of reshaped image> --in_channels <set to 13 for using all channels or 3 for using RGB input> --seed <for reproducibility> --model_type <set vanilla_unet if not using multi temporal input else set to mi_unet> --normalizing_type constant --normalizing_factor 4000 --exp_name <for wandb logging> --exp_number <for wandb logging> --loss BCE [--pretrained] [--use_timepoints]"
```
