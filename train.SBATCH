#!/bin/bash

#SBATCH --job-name=SWIN_FPN_MultiImage_Upsampled_Augmentation_MS_repeat3
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



singularity exec --nv --overlay /scratch/kj1447/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/kj1447/gracelab/SwinTransformerPipeline.py --model_dir /scratch/kj1447/gracelab/models/SWIN_FPN_MultiImage_Upsampled_Augmentation_MS_repeat3
 --batch_size 20 --workers 10 --learning_rate 1e-4 --upsampled_mask_size 228 --upsampled_image_size 228 --in_channels 9 --seed 78 --model_type swin --starting_epoch 0 --normalizing_type constant --normalizing_factor 4000 --use_timepoints --learned_upsampling --exp_name SWIN_FPN_MultiImage_Upsampled_Augmentation_MS --exp_number 3"
