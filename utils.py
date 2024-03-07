import argparse
import torch
import random
import os

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
        parser.add_argument("-lkb", "--lookback", default = 10, type = int, help = "Number of epochs to wait before early stopping")
        parser.add_argument("-se", "--starting_epoch", default = 0, type = int, help = "Epoch number for restarting training")
        parser.add_argument("-ui", "--upsampled_image_size", default = 256, type = int, help = "Size of Upsampled Image")
        parser.add_argument("-um", "--upsampled_mask_size", default = 256, type = int, help = "Size of Upsampled Mask")
        parser.add_argument("-mt","--model_type", default='vanilla_unet', help='type of model')
        parser.add_argument("-ic","--in_channels", default=13, type = int, help='num channels to use')
        parser.add_argument("-ut","--use_timepoints", action="store_true", help='use time channel (true/false)')
        parser.add_argument("-p", "--pretrained",  action="store_true",  help = "Use Pretrained Model")
        parser.add_argument("-cp", "--checkpoint", help = "Path for checkpoint")
        parser.add_argument("-nf","--normalizing_factor", default=4000, type = int, help='Normalizing Factor for images')
        args = parser.parse_args()
    
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
        args['normalizing_factor'] = 4000
        args['model_type']='vanilla_unet'
        args['pretrained']=False
        args['checkpoint']=None
        args['use_timepoints']=False
        args = dotdict(args)
    
    print(args)
    return args


def set_seed(seed):
    #setting seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    