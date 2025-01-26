# Substation segmentation
The aim of this project is to build a computer vision model for segmenting substations in Sentinel 2 satellite imagery. This directory contains code to train and evaluate multiple kinds of segmentation models. We experiment with different kinds of architectures, losses and training strategies. The associated paper is [here](https://arxiv.org/abs/2409.17363) and the dataset is [here](https://huggingface.co/datasets/neurograce/SubstationDataset) 

## The directory structure:
- dataset
  - substation
    - image_stack
    - mask
    - negatives
      - image_stack
      - mask
    - four_or_more_timepoints.pkl
  - PhilEO-downstream/processed_dataset
    - train
      - images
      - building_mask
      - road_mask
      - lc_mask
    - test
    - val
- dataloader.py - creates dataloader for the training
- models.py - creates and instantiates different kinds of models
- utils.py - stores the helped functions
- train.py - training script 
- inference.ipynb - script to run inference from trained models on images
   
## Dataset Details:
**Substation Dataset**
- dataset/image_stack and dataset/mask folders contain all the images and masks respectively.
- There are a a total of 26522 images-mask pairs stored as numpy files.
- Each image is multi-temporal and contains multiple shots taken at the same place during different revisits. Majority of the files contains images from 5 revisits.
- Each image is multi-spectral and  contains 13 channels
- Spatial Resolution of each image is 228*228.
- You can download the dataset from here - [images](https://urldefense.proofpoint.com/v2/url?u=https-3A__storage.googleapis.com_tz-2Dml-2Dpublic_substation-2Dover-2D10km2-2Dcsv-2Dmain-2D444e360fd2b6444b9018d509d0e4f36e_image-5Fstack.tar.gz&d=DwMFaQ&c=slrrB7dE8n7gBJbeO0g-IQ&r=ypwhORbsf5rB8FTl-SAxjfN_U0jrVqx6UDyBtJHbKQY&m=-2QXCp-gZof5HwBsLg7VwQD-pnLedAo09YCzdDCUTqCI-0t789z0-HhhgwVbYtX7&s=zMCjuqjPMHRz5jeEWLCEufHvWxRPdlHEbPnUE7kXPrc&e=) and [masks](https://urldefense.proofpoint.com/v2/url?u=https-3A__storage.googleapis.com_tz-2Dml-2Dpublic_substation-2Dover-2D10km2-2Dcsv-2Dmain-2D444e360fd2b6444b9018d509d0e4f36e_mask.tar.gz&d=DwMFaQ&c=slrrB7dE8n7gBJbeO0g-IQ&r=ypwhORbsf5rB8FTl-SAxjfN_U0jrVqx6UDyBtJHbKQY&m=-2QXCp-gZof5HwBsLg7VwQD-pnLedAo09YCzdDCUTqCI-0t789z0-HhhgwVbYtX7&s=nHMdYvxKmzwAdT2lOPoQ7-NEfjsOjAm00kHOcwC_AmU&e=)
- The negatives comprising of global random images can be found here - [images](https://urldefense.proofpoint.com/v2/url?u=https-3A__storage.googleapis.com_tz-2Dml-2Dpublic_random-2Dsample-2Dcsv-2Dmain-2Da15f73c0e4b94102be4b2aea3b8ef80c_image-5Fstack.tar.gz&d=DwMFaQ&c=slrrB7dE8n7gBJbeO0g-IQ&r=ypwhORbsf5rB8FTl-SAxjfN_U0jrVqx6UDyBtJHbKQY&m=-2QXCp-gZof5HwBsLg7VwQD-pnLedAo09YCzdDCUTqCI-0t789z0-HhhgwVbYtX7&s=z2YEwx2lSeF-aOQSzkZ_XuQ0qyqkukW7n47N8luln5M&e=) and [masks](https://urldefense.proofpoint.com/v2/url?u=https-3A__storage.googleapis.com_tz-2Dml-2Dpublic_random-2Dsample-2Dcsv-2Dmain-2Da15f73c0e4b94102be4b2aea3b8ef80c_mask.tar.gz&d=DwMFaQ&c=slrrB7dE8n7gBJbeO0g-IQ&r=ypwhORbsf5rB8FTl-SAxjfN_U0jrVqx6UDyBtJHbKQY&m=-2QXCp-gZof5HwBsLg7VwQD-pnLedAo09YCzdDCUTqCI-0t789z0-HhhgwVbYtX7&s=LgGOFb6qIn0SNwbG4DTEunWbIgcHdnW2PIWCMvlKboY&e=)  
- Sample image and mask pair are given below -
<p align="center" width="100%">
  <img width="500" src="https://github.com/Lindsay-Lab/substation-seg/blob/main/artifacts/example_input.png">
</p>

**PhilEO Dataset**

- You can find more information about this dataset on [huggingface](https://huggingface.co/datasets/PhilEO-community/PhilEO-downstream)
- Sample image and building mask pair are given below -
<p align="center" width="100%">
  <img width="500" src="https://github.com/Lindsay-Lab/substation-seg/blob/main/artifacts/phileo_example_input_building.png">
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


Sample outputs from our best models areprovided below - 

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
For questions and queries please reach out to [Kartik Jindgar](mailto:kartik.jindgar@nyu.edu) and [Grace Lindsay](mailto:grace.lindsay@nyu.edu)
