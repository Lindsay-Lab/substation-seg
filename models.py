import segmentation_models_pytorch as smp
from torchvision.models._api import WeightsEnum

def setup_model(kind, pretrained, pretrained_weights, resume, checkpoint_path=None):
    
    if kind == 'vanila_unet':        
        model = smp.Unet(
            encoder_name="resnet18",       
            encoder_weights=None,     
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        #     activation = 'sigmoid'     # because using sigmoid_focal_loss
        )
    
    
    if pretrained: 
        assert pretrained_weights is not None
        if isinstance(pretrained_weights, WeightsEnum):
            state_dict = pretrained_weights.get_state_dict(progress=True)
            model.encoder.load_state_dict(state_dict)
     
    if resume:
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        
    return model