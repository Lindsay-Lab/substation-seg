#!/bin/bash

#SBATCH --job-name=imagenet_vit_base_mt_zscore_mt_non_pretrained_repeat3
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G  
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:h100:1
#SBATCH --time=36:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=kj1447@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/kj1447/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/kj1447/gracelab/train.py --model_dir /scratch/kj1447/gracelab/models/imagenet_vit_base_mt_zscore_mt_non_pretrained_repeat2 --batch_size 16 --workers 16 --learning_rate 1e-6 --upsampled_mask_size 224 --upsampled_image_size 224 --in_channels 3 --seed 169 --model_type vit_imagenet --vit_size base --starting_epoch 0 --normalizing_type zscore --exp_name imagenet_vit_base_mt_zscore_mt_non_pretrained --exp_number 3 --loss BCE --type_of_model classification --use_timepoints"
