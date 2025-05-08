import torch
from torch import nn
from networks.classification.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152


def load_original_net(model_deep=50, pretrained=False, in_channels=3, num_classes=1000, ckpt_path='',
                         need_features=True):
    if model_deep == 18:
        expansion = 1
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [64, 64, 64],
                            'layer2': [128, 32, 32],
                            'layer3': [256, 16, 16],
                            'layer4': [512, 8, 8]}
    elif model_deep == 34:
        expansion = 1
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [64, 64, 64],
                            'layer2': [128, 32, 32],
                            'layer3': [256, 16, 16],
                            'layer4': [512, 8, 8]}
    elif model_deep == 50:
        expansion = 4
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    elif model_deep == 101:
        expansion = 4
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    elif model_deep == 152:
        expansion = 4
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    else:
        raise ValueError(f'ResNet model deep: {model_deep} is not supported')
    original_net = eval(
        f'resnet{model_deep}(weights={pretrained}, in_channels={in_channels}, num_classes={num_classes}, ckpt_path="{ckpt_path}", need_features={need_features})')
    layer_shape_dict['head'] = [num_classes]
    return original_net, expansion, layer_shape_dict


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class FeatureSqueeze(nn.Module):
    def __init__(self, features_dim, reduction=16):
        super(FeatureSqueeze, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if features_dim // reduction == 0:
            reduction = features_dim  # X -> 1 -> X
            # reduction = 2

        self.fc = nn.Sequential(
            nn.Linear(features_dim, features_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(features_dim // reduction, features_dim, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
        )

        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc(self.avg_pool(x).squeeze(2).squeeze(2)) + self.fc(self.max_pool(x).squeeze(2).squeeze(2))
        # x = self.relu(x)
        # x = self.dropout(x)
        return x


class TMSL(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(TMSL, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, X):
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        evidence_a = evidences[0]
        for i in range(1, self.num_views):
            evidence_a = (evidences[i] + evidence_a) / 2
        return evidences, evidence_a


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h


class ResNet_CHW_SAFS_TMSL(nn.Module):
    def __init__(self, model_deep=50, pretrained=False, in_channels=3, num_classes=1000, ckpt_path='',
                 need_layer_name_list=None):
        super(ResNet_CHW_SAFS_TMSL, self).__init__()

        if need_layer_name_list is None:
            need_layer_name_list = ['layer1', 'layer2', 'layer3', 'layer4']
        self.need_layer_name_list = need_layer_name_list
        print(f'need_layer_name_list: {self.need_layer_name_list}')

        # Load the weights from the original ResNet model
        self.original_net, self.expansion, self.layer_shape_dict = load_original_net(model_deep,
                                                                                           pretrained,
                                                                                           in_channels=in_channels,
                                                                                           num_classes=num_classes,
                                                                                           ckpt_path=ckpt_path,
                                                                                           need_features=True)

        for p in self.original_net.parameters():  # Freezing backbone
            p.requires_grad = False

        self.MSCB_list = nn.ModuleList()  # Multi-scale feature compression block list
        need_dim_list = []

        axis_reduction = 16
        for layer_name in self.need_layer_name_list:
            if layer_name == 'head':
                continue

            layer_shape = self.layer_shape_dict[layer_name]
            CHW_dim = sum(layer_shape)

            C_sequential = nn.Sequential(
                SpatialAttention(kernel_size=7),
                FeatureSqueeze(features_dim=layer_shape[0], reduction=axis_reduction)
            )
            H_sequential = nn.Sequential(
                SpatialAttention(kernel_size=7),
                FeatureSqueeze(features_dim=layer_shape[1], reduction=axis_reduction)
            )
            W_sequential = nn.Sequential(
                SpatialAttention(kernel_size=7),
                FeatureSqueeze(features_dim=layer_shape[2], reduction=axis_reduction)
            )

            need_dim_list.append(CHW_dim)

            self.MSCB_list.append(nn.ModuleList([C_sequential, H_sequential, W_sequential]))

        if 'head' in self.need_layer_name_list:
            need_dim_list.append(num_classes)
            self.head_relu = nn.ReLU(inplace=True)

        self.concat_relu = nn.ReLU(inplace=True)
        self.concat_dropout = nn.Dropout(0.3)

        num_scale = len(need_dim_list)
        EN_dim_list = []  # Evidence networks dims, shape: (num_scale, num_EN_layers)
        for first_dim in need_dim_list:
            EN_dim_list.append([first_dim])
        self.TMSL = TMSL(num_views=num_scale, dims=EN_dim_list, num_classes=num_classes)

    def forward(self, x):
        out, feature_list = self.original_net(x)

        cls_features = []
        for layer_name in self.need_layer_name_list:
            if layer_name == 'head':
                continue
            cls_features.append(feature_list[layer_name])

        multi_scale_list = []
        for i in range(len(cls_features)):
            C_MSCB = self.MSCB_list[i][0](cls_features[i])

            H_MSCB = self.MSCB_list[i][1](cls_features[i].permute(0, 2, 1, 3))

            W_MSCB = self.MSCB_list[i][2](cls_features[i].permute(0, 3, 1, 2))

            # multi_scale_list.append(self.concat_dropout(self.concat_relu(torch.cat([C_MSCB, H_MSCB, W_MSCB], dim=1))))
            # multi_scale_list.append(self.concat_dropout(torch.cat([C_MSCB, H_MSCB, W_MSCB], dim=1)))

            cat_features = torch.cat([C_MSCB, H_MSCB, W_MSCB], dim=1)
            cat_features = self.concat_relu(cat_features)
            cat_features = self.concat_dropout(cat_features)
            multi_scale_list.append(cat_features)
            

        # if self.need_fusion_multi_scale:
        #     multi_scale_list.append(torch.cat(multi_scale_list, dim=1))

        if 'head' in self.need_layer_name_list:
            multi_scale_list.append(self.head_relu(out))
            # multi_scale_list.append(out)

        evidences, evidence_a = self.TMSL(multi_scale_list)
        return evidences, evidence_a, out

