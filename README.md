# Substation segmentation
The aim of this project is to build a computer vision model for segmenting substations in Sentinel 2 satellite imagery. This directory contains code to train and evaluate multiple kinds of segmentation models. We experiment with different kinds of architectures, losses and training strategies. 

## The directory structure:
- dataset
  - image_stack
  - mask
- dataloader.py - creates dataloader for the training
- models.py - creates and instantiates different kinds of models
- utils.py - stores the helped functions
- train.py - training script 
- inference.py - script to run inference from trained models on images
   
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

> python3 Sentinel.py --model_dir <directory to save models> --batch_size <batch size for dataloader> --workers <number of parallel workers> --learning_rate 1e-3 --upsampled_mask_size <size of reshaped mask> --upsampled_image_size <size of reshaped image> --in_channels <set to 13 for using all channels or 3 for using RGB input> --seed <for reproducibility> --model_type <set vanilla_unet if not using multi temporal input else set to mi_unet> --normalizing_type constant --normalizing_factor 4000 --exp_name <for wandb logging> --exp_number <for wandb logging> --loss BCE --pretrained [add this to use pretrained encoder] --use_timepoints [add this to enable multi-temporal input]

For Training SWIN model- 

> python3 SwinTransformerPipeline.py --model_dir <directory to save models> --batch_size <batch size for dataloader> --workers <number of parallel workers> --learning_rate 5e-4 --upsampled_mask_size <size of reshaped mask> --upsampled_image_size <size of reshaped image> --in_channels <set to 13 for using all channels or 3 for using RGB input> --seed <for reproducibility> --model_type swin --normalizing_type constant --normalizing_factor 4000 --exp_name <for wandb logging> --exp_number <for wandb logging> --loss BCE --pretrained [add this to use pretrained encoder] --use_timepoints [add this to enable multi-temporal input] --learned_upsampling [to append learnable Up-Conv layers at the end of FPN network to upsample Output Mask to Input Image size]

## Training Curves and Sample Outputs
We train all models by minimizing Per-Pixel Binary Cross Entropy Loss and compute Intersection over Union(IoU) to test models. We achieve an impressive IoU score of 58% on test data using the SWIN Model. The loss and IoU curves are given below - 
<p align="center" width="100%">
  <img width="40%" src="https://github.com/Lindsay-Lab/substation-seg/blob/main/artifacts/lossv_sepoch.png">
  <img width="40%" src="https://github.com/Lindsay-Lab/substation-seg/blob/main/artifacts/val_iou_vs_epoch (1).png">
</p>


Sample outputs from our best models is provided below - 

<p align="center" width="100%">
  <img width="75%" src="https://github.com/Lindsay-Lab/substation-seg/blob/main/artifacts/example_output.png">
  <img width="75%" src="https://github.com/Lindsay-Lab/substation-seg/blob/main/artifacts/example_output (1).png">
</p>


## References
* Copernicus Sentinel data [2023]. https://scihub.copernicus.eu/
* Open Street Map. https://www.openstreetmap.org/copyright
* Transition Zero. https://www.transitionzero.org/
* Ze Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. 2021.
arXiv: 2103.14030 [cs.CV].
* Muhammed Razzak et al. Multi-Spectral Multi-Image Super-Resolution of Sentinel-2 with
Radiometric Consistency Losses and Its Effect on Building Delineation. 2021. arXiv: 2111.03231
[eess.IV].
* Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for
Biomedical Image Segmentation. 2015. arXiv: 1505. 04597 [cs.CV].
* Adam J. Stewart et al. “TorchGeo: Deep Learning With Geospatial Data”. In: Proceedings of the
30th International Conference on Advances in Geographic Information Systems. SIGSPATIAL ’22.
Seattle, Washington: Association for Computing Machinery, Nov. 2022, pp. 1–12. doi:
10.1145/3557915.3560953. url: https://dl.acm.org/doi/10.1145/3557915.3560953.
* Piper Wolters, Favyen Bastani, and Aniruddha Kembhavi. Zooming Out on Zooming In:
Advancing Super-Resolution for Remote Sensing. 2023. arXiv:2311.18082 [cs.CV].


## Contact
For questions and queries please reach out to Kartik Jindgar <kartik.jindgar@nyu.edu> and Grace Lindsay <grace.lindsay@nyu.edu>
