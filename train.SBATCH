#!/bin/bash

#SBATCH --job-name=cropped_sentinel_resnet18_augmentation_focal_75_pertimepoint_4k_og
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G  
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
#SBATCH --time=8:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=kj1447@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/kj1447/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/kj1447/gracelab/CroppedSegmentation.py --model_dir cropped_sentinel_resnet18_augmentation_focal_75_pertimepoint_4k --loss FOCAL --epochs 200 --batch_size 128 --workers 16 --lookback 15 --pretrained"

