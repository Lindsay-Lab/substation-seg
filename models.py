import segmentation_models_pytorch as smp
from torchvision.models._api import WeightsEnum
from torch import nn
import torch 
import satlaspretrain_models
import torchgeo
import timm
import segmentation_models_pytorch as smp
from torchvision.ops import sigmoid_focal_loss

class SwinWithUpSample(nn.Module):
    
    def __init__(self, fpn_model, args):
        super(SwinWithUpSample, self).__init__()
        self.args = args
        self.mid_channels = 128
        
        if self.args.type_of_model == 'classification':
            self.num_outputs=2
        else:
            self.num_outputs = 1
            
        self.fpn_model = fpn_model
        self.upsampler = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.mid_channels, self.mid_channels // 2, kernel_size=2, stride=2),
            nn.Conv2d(self.mid_channels // 2, self.mid_channels // 2, kernel_size=3, padding=1),
            #nn.BatchNorm2d(self.mid_channels // 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.mid_channels // 2, self.mid_channels // 4, kernel_size=2, stride=2),
            nn.Conv2d(self.mid_channels // 4, self.mid_channels // 4, kernel_size=3, padding=1),
            #nn.BatchNorm2d(self.mid_channels // 4),
            nn.ReLU(inplace=True),

            torch.nn.Conv2d(self.mid_channels//4, self.num_outputs, 3, padding=1),
        )
                
        def classification_loss_func(logits, targets):
            targets = targets.argmax(dim=1)
            return torch.nn.functional.cross_entropy(logits, targets, reduction='none')[:, None, :, :]

        def regression_loss_func(logits, targets):
            return torch.nn.functional.mse_loss(logits, targets, reduction='none')
        
        if self.args.type_of_model == 'classification':
            self.loss_func = classification_loss_func
        else:
            self.loss_func = regression_loss_func
        
        
    def forward(self, image, target=None):
        loss=None
        feature = self.fpn_model(image)
        raw_outputs = self.upsampler(feature[0])
        
        if self.args.type_of_model == 'classification':
            outputs = torch.nn.functional.softmax(raw_outputs, dim=1)
        else :
            outputs = raw_outputs
        
        if target is not None:
            loss = self.loss_func(raw_outputs, target)
            loss = loss.mean()
        return outputs, loss

class MiUnet(nn.Module):
    def __init__(self, unet, loss_fn, args):
        super(MiUnet,self).__init__()
        self.encoder = unet.encoder
        self.decoder = unet.decoder
        self.segmentation_head = unet.segmentation_head
        # self.num_images_per_timepoint = 3  
        self.loss_fn = loss_fn
        self.args = args
        
    def forward(self, image, target):
        bs, cxi, w, h = image.shape
        num_timepoints = cxi// self.args.in_channels
            
        image = torch.reshape(image, (bs, num_timepoints, -1, w,h))
        bs, t, c, w, h = image.shape
        image = torch.reshape(image, (-1,c,w,h))
        temp = self.encoder(image)
        features = []
        for i in range(len(temp)):
            bsxi, c, w, h = temp[i].shape
            x=torch.reshape(temp[i],(-1,num_timepoints,c, w, h)).permute(1,0,2,3,4) #t,bs,c,w,h
            x = torch.amax(x, 0) #or use this -> x = torch.maximum(torch.maximum(torch.maximum(x[0],x[1]),x[2]), x[3])
            # x = torch.mean(x, 0) #average pooling of multiple timepoints
            features.append(x)
        decoder_output = self.decoder(*features)
        raw_outputs = self.segmentation_head(decoder_output)
        if self.args.loss == 'FOCAL':
            loss = self.loss_fn(raw_outputs, target, alpha = args.alpha, reduction = 'mean')
        else :
            loss = self.loss_fn(raw_outputs, target)
        
        if self.args.type_of_model == 'classification':
            outputs = torch.nn.functional.sigmoid(raw_outputs)
        else:
            outputs = raw_outputs
            
        return outputs, loss        

class ViT(nn.Module):
    def __init__(self,encoder, loss_fn, args, embedding_dim=384, use_timepoints=False):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.output_embed_dim = 1
        self.use_timepoints = use_timepoints
        # self.num_timepoints = 3
        num_convs=4
        num_convs_per_upscale=1
        self.loss_fn = loss_fn
        self.args = args
        
        self.channels = [self.embedding_dim // (2 ** i) for i in range(num_convs)]
        self.channels = [self.embedding_dim] + self.channels

        
        def _build_upscale_block(channels_in, channels_out):
            
            conv_kernel_size = 3
            conv_padding = 1
            kernel_size = 2
            stride = 2
            dilation = 1
            padding = 0
            output_padding = 0
            
            layers = []
            layers.append(nn.ConvTranspose2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ))

            layers += [nn.Sequential(
                      nn.Conv2d(channels_out,
                      channels_out,
                      kernel_size=conv_kernel_size,
                      padding=conv_padding),
                      nn.BatchNorm2d(channels_out),
                      nn.Dropout(),
                      nn.ReLU()) for _ in range(num_convs_per_upscale)]

            return nn.Sequential(*layers)

        self.layers = nn.ModuleList([
            _build_upscale_block(self.channels[i], self.channels[i+1])
            for i in range(len(self.channels) - 1)
        ])

        self.conv = nn.Conv2d(self.channels[-1], self.output_embed_dim, kernel_size = 3, padding=1)

    def forward(self, image, target):
        bs, cxt, h, w = image.shape
        num_timepoints = cxt// self.args.in_channels
        if self.use_timepoints:
            image = image.reshape(bs, num_timepoints, -1, h, w)
            b, t, c, h, w = image.shape
            image = image.reshape(-1, c, h, w)
            
        feature = self.encoder.forward_features(image)
        if self.use_timepoints:
            temporal_features = feature.reshape(-1, num_timepoints, feature.shape[1], feature.shape[2]).permute(1,0,2,3)
            feature = torch.amax(temporal_features, dim = 0)
        feature = feature[:,1:,:].view(-1,14,14,self.embedding_dim).permute(0,3,1,2)
        for layer in self.layers:
            feature = layer(feature)
        raw_outputs = self.conv(feature)
        
        if self.args.loss == 'FOCAL':
            loss = self.loss_fn(raw_outputs, target, alpha = args.alpha, reduction = 'mean')
        else :
            loss = self.loss_fn(raw_outputs, target)
        
        if self.args.type_of_model == 'classification':
            outputs = torch.nn.functional.sigmoid(raw_outputs)
        else: 
            outputs = raw_outputs
        
        return outputs, loss

class VanillaUnet(nn.Module):
    def __init__(self, unet, loss_fn, args):
        super().__init__()
        self.unet = unet
        self.loss_fn = loss_fn
        self.args = args
        
    def forward(self, image, target):
        raw_outputs = self.unet(image) 
        if self.args.loss == 'FOCAL':
            loss = self.loss_fn(raw_outputs, target, alpha = args.alpha, reduction = 'mean')
        else :
            loss = self.loss_fn(raw_outputs, target)
        if self.args.type_of_model == 'classification':
            outputs = torch.nn.functional.sigmoid(raw_outputs)
        else:
            outputs = raw_outputs
        return outputs, loss
        
        
def update_layers(model):
    children = list(model.children())
    if len(children)==0:
        if isinstance(model, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(model.weight.data)
    else:
        for c in children:
            update_layers(c)
            
def setup_loss_func(args):
        if args.model_type != 'swin' :
            if args.loss == 'BCE':
                criterion = nn.BCEWithLogitsLoss()
                loss_fn = lambda output, target : criterion(output, target)
                
            elif args.loss == 'DICE':
                criterion = smp.losses.DiceLoss(mode ='binary')
                loss_fn = lambda output, target : criterion(output, target)
            
            elif args.loss == 'FOCAL':
                loss_fn = sigmoid_focal_loss

            elif args.loss == 'MSE':
                criterion = nn.MSELoss()
                loss_fn = lambda output, target: criterion(output, target)
        else:
            loss_fn = None
        return loss_fn

            
def setup_model(args):
    
    loss_fn = setup_loss_func(args)
    
            
    #Setting Up Model
    if args.model_type == 'vanilla_unet':        
        unet = smp.Unet(
            encoder_name="resnet50",       
            encoder_weights=None,     
            in_channels=args.in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        #     activation = 'sigmoid'     # because using sigmoid_focal_loss
        )
        model = VanillaUnet(unet, loss_fn, args)
    
    elif args.model_type == 'mi_unet':
        unet = smp.Unet(
            encoder_name="resnet50",       
            encoder_weights=None,     
            in_channels=args.in_channels,                  
            classes=1,
        )
        model = MiUnet(unet, loss_fn, args)
        
    elif args.model_type == 'modified_unet':
        
        class ModifiedUNET(nn.Module):
            def __init__(self):
                super(ModifiedUNET, self).__init__()
                self.unet = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels = args.in_channels, classes=1)
                self.unet.decoder.blocks[3].conv1[0] = nn.Conv2d(128, 32, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.decoder.blocks[3].conv2[0] = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.decoder.blocks[4].conv1[0] = nn.Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.decoder.blocks[4].conv2[0] = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), bias=False)
                self.unet.segmentation_head[0] = nn.Conv2d(16, 1, kernel_size=(5,5), stride = (1,1)) #squeeze output 

            def forward(self, x): 
                x = self.unet(x)
                return x 
        
        model = ModifiedUNET()
        
    elif args.model_type == 'swin':
        if args.learned_upsampling:
            #MODEL with learned upsampling
            weights_manager = satlaspretrain_models.Weights()
            fpn_model = weights_manager.get_pretrained_model(args.pretrained_weights, fpn = True,)
            model = SwinWithUpSample(fpn_model,args)
        else:
            weights_manager = satlaspretrain_models.Weights()
            model = weights_manager.get_pretrained_model(args.pretrained_weights, fpn = True, head = satlaspretrain_models.Head.BINSEGMENT, num_categories = 2)
    
    elif args.model_type == 'vit_torchgeo':
        vit_encoder = torchgeo.models.vit_small_patch16_224(args.pretrained_weights)
        model = ViT(vit_encoder,  loss_fn, args, args.embedding_size, use_timepoints = args.use_timepoints)
        
    elif args.model_type == 'vit_imagenet':
        vit_encoder = timm.create_model(args.pretrained_weights, pretrained = args.pretrained)
        model = ViT(vit_encoder, loss_fn, args, args.embedding_size, use_timepoints = args.use_timepoints)
        
    else:
        raise Exception("Incorrect Model Selected")
    
    #Loading Pretrained Weights
    if args.pretrained: 
        #assert pretrained_weights is not None
        if isinstance(args.pretrained_weights, WeightsEnum):
            state_dict = args.pretrained_weights.get_state_dict(progress=True)
            if args.model_type == 'modified_unet':
                model.unet.encoder.load_state_dict(state_dict)
            elif args.model_type == 'vanilla_unet':
                model.unet.encoder.load_state_dict(state_dict)
            elif args.model_type == 'mi_unet': 
                model.encoder.load_state_dict(state_dict)
            else:
                raise Exception("Incorrect combination of weights and model")
    else:
        if args.model_type =='swin':
            #Randomly initializing weights for SWIN model
            update_layers(model)
    
    #Loading checkpoints            
    if args.resume_training:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")    
        else:
            device = torch.device("cpu")
        model = model.to(device)
        checkpoint = torch.load(args.checkpoint,  map_location=device)
        model.load_state_dict(checkpoint)
        
    return model
