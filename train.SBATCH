#!/bin/bash

#SBATCH --job-name=swin_random_img_geoaug_ms_repeat3
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G  
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=kj1447@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/kj1447/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/kj1447/gracelab/train.py --dataset substation --model_dir /scratch/kj1447/gracelab/models/swin_random_img_geoaug_ms_repeat3 --batch_size 16 --workers 16 --learning_rate 1e-4 --upsampled_mask_size 228 --upsampled_image_size 228 --in_channels 9 --seed 78 --model_type swin --starting_epoch 0 --normalizing_type constant --normalizing_factor 4000 --exp_name swin_random_img_geoaug_ms --exp_number 3 --loss BCE --type_of_model classification --learned_upsampling --timepoint_aggregation random --pretrained"
