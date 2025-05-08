from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "get_resnet_layer_shape_dict"
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            in_channels: int = 3,
            need_features: bool = False,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channels = in_channels
        self.need_features = need_features
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        need_features_dict = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.need_features:
            need_features_dict['stem'] = x

        x = self.layer1(x)
        if self.need_features:
            need_features_dict['layer1'] = x
        x = self.layer2(x)
        if self.need_features:
            need_features_dict['layer2'] = x
        x = self.layer3(x)
        if self.need_features:
            need_features_dict['layer3'] = x
        x = self.layer4(x)
        if self.need_features:
            need_features_dict['layer4'] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.need_features:
            return x, need_features_dict
        else:
            return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def load_pretrained_mode(model, ckpt_path):
    if ckpt_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(ckpt_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')

    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']  # model or model_ema
    else:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()
    for k in list(checkpoint_model.keys()):
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    model.load_state_dict(checkpoint_model, strict=False)


def _resnet(block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], weights: bool, ckpt_path: str,
            **kwargs: Any, ) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    if weights:
        load_pretrained_mode(model, ckpt_path)

    return model


def resnet18(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
    """
    if ckpt_path == '':
        ckpt_path = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, ckpt_path, **kwargs)


def resnet34(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                }
    """
    if ckpt_path == '':
        ckpt_path = 'https://download.pytorch.org/models/resnet34-b627a593.pth'

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, ckpt_path, **kwargs)


def resnet50(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    v1
    "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
    v2
    "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    """
    if ckpt_path == '':
        # ckpt_path = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'  # v1
        ckpt_path = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'  # v2

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, ckpt_path, **kwargs)


def resnet101(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    v1
    "ImageNet-1K": {
                    "acc@1": 77.374,
                    "acc@5": 93.546,
                }
    v2
    "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 95.780,
                }
    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.


    """
    if ckpt_path == '':
        # ckpt_path = 'https://download.pytorch.org/models/resnet101-63fe2227.pth'  # v1
        ckpt_path = 'https://download.pytorch.org/models/resnet101-cd907fc2.pth'  # v2

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, ckpt_path, **kwargs)


def resnet152(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    v1
    "ImageNet-1K": {
                    "acc@1": 78.312,
                    "acc@5": 94.046,
                }
    v2
    "ImageNet-1K": {
                    "acc@1": 82.284,
                    "acc@5": 96.002,
                }
    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    """
    if ckpt_path == '':
        # ckpt_path = 'https://download.pytorch.org/models/resnet152-394f9c45.pth'  # v1
        ckpt_path = 'https://download.pytorch.org/models/resnet152-f82ba261.pth'  # v2

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, ckpt_path, **kwargs)


def resnext50_32x4d(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    v1
    "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                }
    v2
    "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                }

    """
    if ckpt_path == '':
        # ckpt_path = 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'  # v1
        ckpt_path = 'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth'  # v2

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, ckpt_path, groups=32, width_per_group=4, **kwargs)


def resnext101_32x8d(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    v1
    "ImageNet-1K": {
                    "acc@1": 79.312,
                    "acc@5": 94.526,
                }
    v2
    "ImageNet-1K": {
                    "acc@1": 82.834,
                    "acc@5": 96.228,
                }

    """
    if ckpt_path == '':
        # ckpt_path = 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'  # v1
        ckpt_path = 'https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth'  # v2

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, ckpt_path, groups=32, width_per_group=8, **kwargs)


def resnext101_64x4d(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    "ImageNet-1K": {
                    "acc@1": 83.246,
                    "acc@5": 96.454,
                }

    """
    if ckpt_path == '':
        ckpt_path = 'https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth'

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, ckpt_path, groups=64, width_per_group=4, **kwargs)


def wide_resnet50_2(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    v1
    "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                }
    v2
    "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                }
    """

    if ckpt_path == '':
        # ckpt_path = 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'  # v1
        ckpt_path = 'https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth'  # v2

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, ckpt_path, width_per_group=128, **kwargs)


def wide_resnet101_2(*, weights: bool = None, ckpt_path: str = '', **kwargs: Any) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    v1
    "ImageNet-1K": {
                    "acc@1": 78.848,
                    "acc@5": 94.284,
                }
    v2
    "ImageNet-1K": {
                    "acc@1": 82.510,
                    "acc@5": 96.020,
                }

    """
    if ckpt_path == '':
        # ckpt_path = 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'  # v1
        ckpt_path = 'https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth'  # v2

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, ckpt_path, width_per_group=128, **kwargs)


def get_resnet_layer_shape_dict(model_deep=18):
    if model_deep == 18:
        get_mode_function_name = 'resnet18'
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [64, 64, 64],
                            'layer2': [128, 32, 32],
                            'layer3': [256, 16, 16],
                            'layer4': [512, 8, 8]}
    elif model_deep == 34:
        get_mode_function_name = 'resnet34'
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [64, 64, 64],
                            'layer2': [128, 32, 32],
                            'layer3': [256, 16, 16],
                            'layer4': [512, 8, 8]}
    elif model_deep == 50:
        get_mode_function_name = 'resnet50'
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    elif model_deep == 101:
        get_mode_function_name = 'resnet101'
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [64, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    elif model_deep == 152:
        get_mode_function_name = 'resnet152'
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    else:
        raise ValueError(f'ResNet model deep: {model_deep} is not supported')
    return layer_shape_dict, get_mode_function_name
