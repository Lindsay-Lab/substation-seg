#!/bin/bash

#SBATCH --job-name=Unet_Resnet50_MS_224_MultiImg_repeat3
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G  
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=kj1447@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/kj1447/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/kj1447/gracelab/Sentinel.py --model_dir /scratch/kj1447/gracelab/models/Unet_Resnet50_MS_224_MultiImg_repeat3 --batch_size 20 --workers 10 --learning_rate 1e-3 --upsampled_mask_size 224 --upsampled_image_size 224 --in_channels 13 --seed 14188518 --model_type mi_unet --starting_epoch 0 --normalizing_type constant --normalizing_factor 4000 --exp_name Unet_Resnet50_MS_224_MultiImg --exp_number 3 --loss BCE --use_timepoints"
