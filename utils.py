import argparse
import numpy as np
import torch
import random
import os
from torchgeo.models import ResNet18_Weights,ResNet50_Weights, ViTSmall16_Weights

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def parse_arguments(cmd_flag=False):
    
    if cmd_flag:
        parser = argparse.ArgumentParser(description = "Define parameters like loss function, model_directiory etc")
        parser.add_argument("-d","--data_dir", default = "/scratch/kj1447/gracelab/dataset", help="Data Directory")
        parser.add_argument("-m","--model_dir", help="Model Directory for Saving Checkpoints. A folder with this name would be created within the models folder")
        parser.add_argument("-l","--loss", default = "BCE", help="Can take following values - BCE, FOCAL or DICE")
        parser.add_argument("-e","--epochs", default = 250, type = int, help="Number of epochs")
        parser.add_argument("-bs","--batch_size", default = 32, type=int, help="Batch Size")
        parser.add_argument("-w","--workers", default = 16, type=int, help="Number of Workers")
        parser.add_argument("-tr", "--train_ratio", default = 0.8, type = float, help = "Train-Validation Ratio")
        parser.add_argument("-lr", "--learning_rate", default = 1e-3, type = float, help = "Learning Rate")
        parser.add_argument("-s","--seed", default = 42, type = int, help="Seed for reproducibility")
        parser.add_argument("-r", "--resume_training",  action="store_true",  help = "Resume training from checkpoint")
        parser.add_argument("-lkb", "--lookback", default = 20, type = int, help = "Number of epochs to wait before early stopping")
        parser.add_argument("-se", "--starting_epoch", default = 0, type = int, help = "Epoch number for restarting training")
        parser.add_argument("-ui", "--upsampled_image_size", default = 256, type = int, help = "Size of Upsampled Image")
        parser.add_argument("-um", "--upsampled_mask_size", default = 256, type = int, help = "Size of Upsampled Mask")
        parser.add_argument("-mt","--model_type", default='vanilla_unet', help='type of model')
        parser.add_argument("-ic","--in_channels", default=13, type = int, help='num channels to use')
        parser.add_argument("-ut","--use_timepoints", action="store_true", help='use time channel (true/false)')
        parser.add_argument("-p", "--pretrained",  action="store_true",  help = "Use Pretrained Model")
        parser.add_argument("-cp", "--checkpoint", help = "Path for checkpoint")
        parser.add_argument("-nt","--normalizing_type", default='percentile', help='Type of Normalization Used. Either "percentile" or "constant". If its is percentile, we use 1st and 99th percentile to perform linear scaling. Else a constant range is used.')
        parser.add_argument("-nf","--normalizing_factor", default=4000, type = int, help='Normalizing Factor for images')
        parser.add_argument("-lu", "--learned_upsampling",  action="store_true",  help = "Flag to train Deconvolution Layers on top of Swin Transformer")
        parser.add_argument("-ena", "--exp_name", help = "Experiment Name for wanDB tracking")
        parser.add_argument("-enu", "--exp_number",  default=1,  help = "Experiment Number with same setting for wanDB tracking")
        
        parser.add_argument("-vs","--vit_size", default = 'vit_base', help='vit architecture')
        parser.add_argument("-vp","--vit_pretraining", action = 'store_true', help='pretrained vit or not')
        args = vars(parser.parse_args())
    
    else:       
        # if cmdline arguments can't be passed. eg. when running through a jupyter notebook
        args = {}
        args['data_dir']= "/scratch/kj1447/gracelab/dataset"
        args['model_dir'] = '/scratch/kj1447/gracelab/models/SWIN_FPN_low_LR'
        args['loss'] = "BCE"
        args['epochs'] = 250
        args['batch_size'] = 16
        args['workers'] = 16
        args['train_ratio'] = 0.8
        args['learning_rate'] = 1e-4
        args['lookback'] = 10
        args['seed'] = 42
        args['resume_training'] = False
        args['starting_epoch'] = 0
        args['upsampled_image_size']=256
        args['upsampled_mask_size']=64
        args['in_channels'] = 3
        args['normalizing_type'] = "constant"
        args['normalizing_factor'] = 4000
        args['model_type']='swin'
        args['pretrained']=True
        args['checkpoint']=None
        args['use_timepoints']=True
        args['learned_upsampling']=True
        args['exp_name']="SWIN_MI"
        args['exp_number']=1
    
    args = dotdict(args)
    args = sanity_checks(args)
    print(args)
    return args

def sanity_checks(args):
    
    #defining pretrained weights
    if args.model_type == 'vanilla_unet' or args.model_type =='modified_unet' or args.model_type == 'mi_unet':
        if args.pretrained:
            if args.in_channels == 3:
                args.pretrained_weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
            elif args.in_channels == 13:
                args.pretrained_weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            else:
                assert False 
        else:
            args.pretrained_weights = None
        
    elif args.model_type=='swin':
        if args.in_channels == 3:    
            if args.use_timepoints:
                args.pretrained_weights = "Sentinel2_SwinB_MI_RGB"
            else:
                args.pretrained_weights = "Sentinel2_SwinB_SI_RGB"
        else:
            args.pretrained_weights = "Sentinel2_SwinB_MI_MS"
            # raise Exception("SWIN Backbone not Implemented with Multi Channel input")
    
    elif args.model_type == 'vit_torchgeo':
        args.pretrained_weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO
        args.embedding_size = 384
    
    elif args.model_type == 'vit_imagenet':
        if args.vit_size=='small':
            args.pretrained_weights='vit_small_patch16_224'
            args.embedding_size = 384
        elif args.vit_size == 'base':
            args.pretrained_weights= 'vit_base_patch16_224'
            args.embedding_size = 768
        elif args.vit_size == 'large':
            args.pretrained_weights= 'vit_large_patch16_224'
            args.embedding_size = 1024
            
    if args.normalizing_type == 'percentile':    
        args.normalizing_factor = np.array([[1187.   , 1836.   ,  649.   ],
                                            [ 878.   , 1931.085, 1053.085],
                                            [ 749.   , 1982.51 , 1233.51 ],
                                            [ 478.   , 2287.17 , 1809.17 ],
                                            [ 744.   , 2317.   , 1573.   ],
                                            [1248.   , 3195.   , 1947.   ],
                                            [1389.   , 3853.   , 2464.   ],
                                            [1205.83 , 3840.335, 2634.505],
                                            [1455.   , 4186.   , 2731.   ],
                                            [ 462.   , 1084.   ,  622.   ],
                                            [  10.   ,   16.   ,    6.   ],
                                            [1053.   , 3444.68 , 2391.68 ],
                                            [ 501.   , 2715.   , 2214.   ]])
    
    if args.pretrained:
        assert args.pretrained_weights is not None
    
    if args.resume_training:
        assert args.checkpoint is not None
        assert os.path.exists(args.checkpoint)
    
    if args.model_type=='swin':
        if args.in_channels!=3:
            if args.in_channels !=9:
                raise Warning("Swin Model can take only 9 channles for MS input. Setting input channels to 9.")
                args.in_channels = 9
    elif args.model_type == 'vit_imagenet':
        if args.in_channels != 3:
            raise Warning("Vit Imagenet Model can take only 3 channles. Setting input channels to 3.")
            args.in_channels = 3
    elif args.model_type == 'vit_torchgeo':
        if args.in_channels != 13:
            raise Warning("Vit Torchgeo Model can take only 13 channles. Setting input channels to 13.")
            args.in_channels = 13

    return args
    
    
def set_seed(seed):
    #setting seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    