import torch
from torchvision import models

from networks.classification.ResNet import *
from networks.classification.VanillaNet import *
from networks.classification.SwinTransformer import *
from networks.classification.ResNet_Multi_Scale import ResNet_CHW_SAFS_TMSL
from networks.classification.VanillaNet_Multi_Scale import VanillaNet_CHW_SAFS_TMSL
from networks.classification.SwinTransformer_Multi_Scale import SwinTransformer_CHW_SAFS_TMSL


def net_factory(args):
    net_type = args.model
    net_deep = args.model_deep
    in_channels = args.in_channels
    num_classes = args.num_classes
    ckpt_path = args.ckpt_path
    need_layer_name_list = args.need_layer_name_list

    if (args.pretrained == 1) or (args.pretrained == '1'):
        pretrained = True
    else:
        pretrained = False

    if net_type == 'ResNet':
        if net_deep in [18, 32, 50, 101, 152]:
            net = eval(
                f'resnet{net_deep}(weights={pretrained}, in_channels={in_channels}, num_classes={num_classes}, ckpt_path="{ckpt_path}")')
        else:
            raise ValueError(f'ResNet not support deep [18, 32, 50, 101, 152]: {net_deep}')
    elif net_type == 'ResNet_CHW_SAFS_TMSL':
        net = ResNet_CHW_SAFS_TMSL(model_deep=net_deep, pretrained=True, in_channels=in_channels,
                                   num_classes=num_classes, ckpt_path=ckpt_path,
                                   need_layer_name_list=need_layer_name_list)
    elif net_type == 'VanillaNet':
        if net_deep in [5, 6, 7, 8, 9, 10, 11, 12, 13]:
            net = eval(
                f'vanillanet_{net_deep}(in_chans={in_channels}, num_classes={num_classes}, pretrained={pretrained}, ckpt_path="{ckpt_path}")')
        else:
            raise ValueError(f'VanillaNet not support deep [5, 6, 7, 8, 9, 10, 11, 12, 13]: {net_deep}')
    elif net_type == 'VanillaNet_CHW_SAFS_TMSL':
        net = VanillaNet_CHW_SAFS_TMSL(model_deep=net_deep, pretrained=True, in_channels=in_channels,
                                       num_classes=num_classes, ckpt_path=ckpt_path,
                                       need_layer_name_list=need_layer_name_list)
    elif net_type == 'SwinTransformer':
        if net_deep == 0:
            model_deep = 't'
        elif net_deep == 1:
            model_deep = 's'
        elif net_deep == 2:
            model_deep = 'b'
        else:
            raise ValueError(f'Swin Transfoemer not support deep [0, 1, 2]: {net_deep}')
        net = eval(
            f'swin_{model_deep}(in_channels={in_channels}, num_classes={num_classes}, weights={pretrained}, ckpt_path="{ckpt_path}")')
    elif net_type == 'SwinTransformer_CHW_SAFS_TMSL':
        net = SwinTransformer_CHW_SAFS_TMSL(model_deep=net_deep, pretrained=True, in_channels=in_channels,
                                            num_classes=num_classes, ckpt_path=ckpt_path,
                                            need_layer_name_list=need_layer_name_list)
    else:
        raise ValueError(f'{net_type} is not supported')

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    return net