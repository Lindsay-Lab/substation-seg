import segmentation_models_pytorch as smp
from torchvision.models._api import WeightsEnum
from torch import nn
import torch 

def setup_model(kind, in_channels, pretrained, pretrained_weights, resume, checkpoint_path=None):
    
    if kind == 'vanilla_unet':        
        model = smp.Unet(
            encoder_name="resnet18",       
            encoder_weights=None,     
            in_channels=in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        #     activation = 'sigmoid'     # because using sigmoid_focal_loss
        )
    
    elif kind == 'modified_unet':
        
        class ModifiedUNET(nn.Module):
            def __init__(self):
                super(ModifiedUNET, self).__init__()
                self.unet = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=in_channels, classes=1)
                self.unet.decoder.blocks[3].conv1[0] = nn.Conv2d(128, 32, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.decoder.blocks[3].conv2[0] = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.decoder.blocks[4].conv1[0] = nn.Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.decoder.blocks[4].conv2[0] = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.segmentation_head[0] = nn.Conv2d(16, 1, kernel_size=(5,5), stride = (1,1)) #squeeze output 

            def forward(self, x): 
                x = self.unet(x)
                return x 
        
        model = ModifiedUNET()
    else:
        raise Exception("Incorrect Model Selected")
    if pretrained: 
        assert pretrained_weights is not None
        if isinstance(pretrained_weights, WeightsEnum):
            state_dict = pretrained_weights.get_state_dict(progress=True)
            if kind == 'modified_unet':
                model.unet.encoder.load_state_dict(state_dict)
            elif kind == 'vanilla_unet': 
                model.encoder.load_state_dict(state_dict)
     
    if resume:
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        
    return model
