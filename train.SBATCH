#!/bin/bash

#SBATCH --job-name=phileo_unet_output_space_wdecay_repeat4
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G  
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=kj1447@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/kj1447/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/kj1447/gracelab/train.py --dataset phileo --model_dir /scratch/kj1447/gracelab/models/phileo_unet_output_space_wdecay_repeat4 --batch_size 16 --workers 16 --learning_rate 1e-6 --upsampled_mask_size 224 --upsampled_image_size 224 --in_channels 13 --seed 169 --model_type mi_unet_output_space --starting_epoch 0 --normalizing_type constant --normalizing_factor 4000 --exp_name phileo_unet_output_space_wdecay --exp_number 4 --loss MSE --type_of_model regression --use_timepoints --timepoint_aggregation concat --pretrained --learned_upsampling"
