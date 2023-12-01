import argparse
import torch
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Define parameters like loss function, model_directiory etc")

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
    parser.add_argument("-ic","--in_channels", default=13, help='num channels to use')
    parser.add_argument("-ut","--use_timepoints", default=False, help='use time channel (true/false)')
    args = parser.parse_args()

    print(args)
    
    return args


def set_seed(seed):
    #setting seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
