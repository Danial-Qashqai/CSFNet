import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def CSFNet(version, pretrain, backbone_path ,dataset, num_classes):

      if version == "CSFNet-1":
                   backbone="STDCNet813"

      if version == "CSFNet-2":
                   backbone="STDCNet1446"

      return Network(num_class=num_classes, pretrain=pretrain , backbone_path=backbone_path ,backbone=backbone,dataset=dataset)

class Network(nn.Module):
    def __init__(self, num_class=19, backbone='STDCNet813', pretrain=None ,backbone_path=None ,dataset ="Cityscapes", act_type='relu'):
        super(Network, self).__init__()

        if dataset == "Cityscapes":
             pool_out=[(16, 32),(8, 16),(4, 8),(2, 4),(4,8)]

        if dataset == "MFNet" or dataset == "ZJU" or dataset == "FMB":
             pool_out = [(16, 24), (8, 12), (4, 6), (2, 3), (5, 5)]

        self.backbone_name = backbone
        decoder_channels = [32, 64, 128, 32, num_class]

        self.encoder = Encoder(backbone, pretrain, backbone_path, dataset, pool_out)
        self.CM = Context_Module(1024, decoder_channels[0], act_type, pooling_size=pool_out[4])
        self.decoder = Decoder(decoder_channels, act_type,pool_out,dataset)

    def forward(self, rgb, depth):
        x1, x3, x4, x5 = self.encoder(rgb, depth)
        x5 = self.CM(x5)
        x = self.decoder(x1, x3, x4, x5)

        return x


class Encoder(nn.Module):
    def __init__(self, backbone, pretrain, backbone_path, dataset, pool_out):
        super(Encoder, self).__init__()

        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.encoder = STDCNet1446(pretrain_model=pretrain, backbone_path=backbone_path ,  dataset=dataset , pool_out=pool_out)

        elif backbone == 'STDCNet813':
            self.encoder = STDCNet813(pretrain_model=pretrain, backbone_path=backbone_path , dataset=dataset , pool_out=pool_out)

    def forward(self, rgb, depth):
        x1, x3, x4, x5 = self.encoder(rgb, depth)
        return x1, x3, x4, x5


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                          'celu': nn.CELU, 'elu': nn.ELU,
                          'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU, 'glu': nn.GLU,
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                          'tanh': nn.Tanh, 'none': nn.Identity,
                          }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


class decoder_fusion(nn.Module):
    def __init__(self, num_channel, hid_channels, pool_size):
        super(decoder_fusion, self).__init__()

        self.attention = fusion_attention(num_channel, hid_channels, pool_size)

    def forward(self, x_high, x_low):
        alpha, beta = self.attention(x_high, x_low)
        x = (alpha * x_high) + (beta * x_low)

        return x


class Decoder(nn.Module):
    def __init__(self, decoder_channels, act_type, pool_out,dataset="ZJU"):
        super(Decoder, self).__init__()
        self.dataset = dataset

        self.CBR1 = ConvBNAct(decoder_channels[0], decoder_channels[0])
        self.CBR2 = ConvBNAct(decoder_channels[0], decoder_channels[1])
        self.CBR3 = ConvBNAct(decoder_channels[1], decoder_channels[2])
        self.CBR4 = ConvBNAct(decoder_channels[2], decoder_channels[3])
        self.CBR5 = ConvBNAct(decoder_channels[3], decoder_channels[4])

        self.D_fusion1 = decoder_fusion(32, 16, pool_out[3])
        self.D_fusion2 = decoder_fusion(64, 32, pool_out[2])
        self.D_fusion3 = decoder_fusion(32, 16, pool_out[0])

    def forward(self, x1, x3, x4, x5):
        x = self.CBR1(x5)

        if self.dataset == "ZJU":
            x = F.interpolate(x, size=(32, 39), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.D_fusion1(x, x4)

        x = self.CBR2(x)

        if self.dataset == "ZJU":
            x = F.interpolate(x, size=(64,77), mode='bilinear', align_corners=True)
        elif self.dataset == "FMB":
            x = F.interpolate(x, size=(75,100), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.D_fusion2(x, x3)

        x = self.CBR3(x)
        x = self.CBR4(x)

        if self.dataset == "ZJU":
            x = F.interpolate(x, size=(256,306), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        x = self.D_fusion3(x, x1)

        x = self.CBR5(x)
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return out


class Context_Module(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, pooling_size):
        super(Context_Module, self).__init__()
        hid_channels = int(in_channels // 4)
        self.act_type = act_type

        self.pool1 = self._make_pool_layer(in_channels, hid_channels, pooling_size)

        self.conv2 = self._make_conv_layer(hid_channels, 64, (1, 4), 1)
        self.conv3 = self._make_conv_layer(hid_channels, 64, (4, 1), 1)

        self.conv = conv3x3(64, out_channels)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type='relu')
        )

    def _make_pool_layer(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            ConvBNAct(in_channels, out_channels, 1, act_type=self.act_type)
        )

    def forward(self, x):
        size = x.size()[2:]

        x = self.pool1(x)

        x1 = F.interpolate(self.conv2(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.conv3(x), size, mode='bilinear', align_corners=True)
        x = self.conv(x1 + x2)

        return x


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2, 2, 2], block_num=4, type="cat", backbone_path=None, dataset = "Cityscapes", pretrain_model=False ,pool_out=None):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck

        self.backbone_path = backbone_path

        self.features = self._make_layers(base, layers, block_num, block)
        self.features_d = self._make_layersd(base, layers, block_num, block, dataset)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        self.x2d = nn.Sequential(self.features_d[:1])
        self.x4d = nn.Sequential(self.features_d[1:2])
        self.x8d = nn.Sequential(self.features_d[2:4])

        self.f1 = encoder_fusion(32, 16, pool_out[0] , skip_conncetion=True, last_fusion=False)
        self.f2 = encoder_fusion(64, 32, pool_out[1] , skip_conncetion=False, last_fusion=False)
        self.f3 = encoder_fusion(256, 128, pool_out[2] , skip_conncetion=True, last_fusion=True)

        self.agent4 = ConvBNAct(512, 32, 1, act_type='relu')
        self.agent3 = ConvBNAct(256, 64, 1, act_type='relu')
        self.agent1 = ConvBNAct(32, 32, 1, act_type='relu')

        if pretrain_model:
            print('use pretrained STDC1: {}'.format(pretrain_model))
            self.init_weight(dataset)

    def init_weight(self, dataset):
        pretrain_dict = torch.load(self.backbone_path )["state_dict"]  # "/kaggle/input/stdc1-back/STDCNet813M_73.91.tar"
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('features.0.conv'):
                    model_dict[k] = v
                    # print('##### ', k)
                    if dataset == "Cityscapes":
                            model_dict[k[:8] + '_d' + k[8:]] = torch.cat((torch.mean(v, 1), torch.mean(v, 1)), dim=1).data. \
                                view_as(state_dict[k.replace('features.0.conv.weight', 'features_d.0.conv.weight')])

                    elif dataset == "ZJU":
                            model_dict[k[:8] + '_d' + k[8:]] = v

                    else:
                            model_dict[k[:8] + '_d' + k[8:]] = torch.mean(v, 1).data. \
                                view_as(state_dict[k.replace('features.0.conv.weight', 'features_d.0.conv.weight')])
                elif k.startswith('features'):
                    model_dict[k] = v
                    model_dict[k[:8] + '_d' + k[8:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def _make_layersd(self, base, layers, block_num, block, dataset):
        features = []

        if dataset== "Cityscapes":
            in_channel=2

        elif dataset== "ZJU":
            in_channel=3

        else:
            in_channel = 1

        features += [ConvX(in_channel, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, rgb, depth):
        rgb = self.x2(rgb)
        depth = self.x2d(depth)
        rgb, depth, s1 = self.f1(rgb, depth)
        x1 = self.agent1(s1)

        rgb = self.x4(rgb)
        depth = self.x4d(depth)
        rgb, depth = self.f2(rgb, depth)

        rgb = self.x8(rgb)
        depth = self.x8d(depth)
        s3 = self.f3(rgb, depth)
        x3 = self.agent3(s3)

        s4 = self.x16(s3)
        x4 = self.agent4(s4)

        x5 = self.x32(s4)

        return x1, x3, x4, x5


class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, type="cat", backbone_path=None,  dataset = "Cityscapes" , pretrain_model=False , pool_out= None):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck

        self.backbone_path = backbone_path

        self.features = self._make_layers(base, layers, block_num, block)
        self.features_d = self._make_layersd(base, layers, block_num, block, dataset)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        self.x2d = nn.Sequential(self.features_d[:1])
        self.x4d = nn.Sequential(self.features_d[1:2])
        self.x8d = nn.Sequential(self.features_d[2:6])

        self.f1 = encoder_fusion(32, 16, pool_out[0] , skip_conncetion=True, last_fusion=False)
        self.f2 = encoder_fusion(64, 32, pool_out[1], skip_conncetion=False, last_fusion=False)
        self.f3 = encoder_fusion(256, 128, pool_out[2] , skip_conncetion=True, last_fusion=True)

        self.agent4 = ConvBNAct(512, 32, 1, act_type='relu')
        self.agent3 = ConvBNAct(256, 64, 1, act_type='relu')
        self.agent1 = ConvBNAct(32, 32, 1, act_type='relu')

        if pretrain_model:
            print('use pretrained STDC2: {}'.format(pretrain_model))
            self.init_weight(dataset)


    def init_weight(self, dataset):
        pretrain_dict = torch.load(self.backbone_path)["state_dict"]    # "/kaggle/input/stdc2-back/STDCNet1446_76.47.tar"
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('features.0.conv'):
                    model_dict[k] = v
                    # print('##### ', k)
                    if dataset == "Cityscapes":
                            model_dict[k[:8] + '_d' + k[8:]] = torch.cat((torch.mean(v, 1), torch.mean(v, 1)), dim=1).data. \
                                view_as(state_dict[k.replace('features.0.conv.weight', 'features_d.0.conv.weight')])

                    elif dataset == "ZJU":
                            model_dict[k[:8] + '_d' + k[8:]] = v

                    else:
                            model_dict[k[:8] + '_d' + k[8:]] = torch.mean(v, 1).data. \
                                view_as(state_dict[k.replace('features.0.conv.weight', 'features_d.0.conv.weight')])
                elif k.startswith('features'):
                    model_dict[k] = v
                    model_dict[k[:8] + '_d' + k[8:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def _make_layersd(self, base, layers, block_num, block, dataset):
        features = []

        if dataset== "Cityscapes":
            in_channel=2

        elif dataset== "ZJU":
            in_channel=3
        else:
            in_channel = 1

        features += [ConvX(in_channel, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, rgb, depth):
        rgb = self.x2(rgb)
        depth = self.x2d(depth)
        rgb, depth, s1 = self.f1(rgb, depth)
        x1 = self.agent1(s1)

        rgb = self.x4(rgb)
        depth = self.x4d(depth)
        rgb, depth = self.f2(rgb, depth)

        rgb = self.x8(rgb)
        depth = self.x8d(depth)
        s3 = self.f3(rgb, depth)
        x3 = self.agent3(s3)

        s4 = self.x16(s3)
        x4 = self.agent4(s4)

        x5 = self.x32(s4)

        return x1, x3, x4, x5

class fusion_attention(nn.Module):
    def __init__(self, num_channel, out_channel, pool_size):
        super(fusion_attention, self).__init__()

        # todo add convolution here
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.conv1 = nn.Conv2d(num_channel, out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channel, num_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.activation = nn.Sigmoid()

    def forward(self, rgb, depth):
        rgb = self.pool(rgb).view((rgb.size()[0], rgb.size()[1], -1))
        depth = self.pool(depth).view((depth.size()[0], depth.size()[1], -1))
        similarity_vector = self.cos(rgb, depth).view(rgb.size()[0], rgb.size()[1], 1, 1)

        output = F.relu(self.bn1(self.conv1(similarity_vector)))
        output = self.conv2(output)
        weight = self.activation(output)
        return weight, 1 - weight


class encoder_fusion(nn.Module):
    def __init__(self, num_channel, hid_channels, pool_size, skip_conncetion, last_fusion):
        super(encoder_fusion, self).__init__()

        self.attention = fusion_attention(num_channel, hid_channels, pool_size)
        self.skip_conncetion = skip_conncetion
        self.last_fusion = last_fusion

    def forward(self, rgb, depth):

        wd, wr = self.attention(rgb, depth)
        w_rgb = rgb.mul(wr)
        w_depth = depth.mul(wd)

        if self.skip_conncetion == True and self.last_fusion == False:
            rgb = rgb + w_depth
            depth = depth + w_rgb
            SC = w_rgb + w_depth
            return rgb, depth, SC

        if self.skip_conncetion == False and self.last_fusion == False:
            rgb = rgb + w_depth
            depth = depth + w_rgb
            return rgb, depth

        if self.skip_conncetion == True and self.last_fusion == True:
            SC = w_rgb + w_depth
            return SC