import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class mlblock(nn.Module):
    def __init__(self, inplane, outplane, p=0.0):
        super(mlblock, self).__init__()
        model = []
        model.append(nn.Linear(inplane, outplane))
        model.append(nn.BatchNorm1d(outplane))
        model.append(nn.ReLU(True))
        model.append(nn.Dropout(p=p, inplace=False))
        self.layer = nn.Sequential(*model)

    def forward(self, x):
        return self.layer(x)


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self,inplane, outplane, kernel_size=3, stride=1, padding=1):
        super(ConvDropoutNormNonlin,self).__init__()
        self.conv = nn.Conv3d(inplane, outplane, kernel_size=kernel_size,stride=stride, padding=padding)
        self.instnorm = nn.InstanceNorm3d(outplane,affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
    def forward(self, x):
        return self.lrelu(self.instnorm(self.conv(x)))

class StackedConvLayers(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3, stride=2, padding=1, p=0.0,is_first=False,is_up=False,is_one=True, is_same_last =True):
        super(StackedConvLayers, self).__init__()
        if is_same_last:
            neark_kernel_size = kernel_size
            near_pad = padding
        else:
            neark_kernel_size = 3
            near_pad = 1

        if not is_up:
            if  is_first:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=1, padding=padding),
                    ConvDropoutNormNonlin(outplane, outplane, kernel_size=kernel_size, stride=1, padding=padding)
                )
            else:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding),
                    ConvDropoutNormNonlin(outplane, outplane, kernel_size=neark_kernel_size, stride=1, padding=near_pad)
                )
        else:
            if is_one:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=1, padding=padding),
                )
            else:
                self.blocks = nn.Sequential(
                    ConvDropoutNormNonlin(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding),
                )
    def forward(self,x):
        return self.blocks(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, kernel=(3, 3, 1), padding=(1, 1, 0), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel, stride=stride,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_down = nn.Conv3d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv3d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # Downsample
        self.downsample = downsample
        self.stride = stride
        self.relu_final = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual
        res = self.relu_final(res)

        return res


class SEResNet(nn.Module):

    def __init__(self, block, layers, int_channel=1, classes=None):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(int_channel, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if self.classes is not None:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(2048, classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] *  m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel=3, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel, padding, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel=kernel, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        feature.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feature.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)
        if self.classes is not None:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        return feature


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm3d(out_channels)

        else:
            bn = nn.Identity()
        super(Conv3dReLU, self).__init__(conv, bn, relu)

class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super(UnetDecoder, self).__init__()
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.center = nn.Identity()
    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool3d(1) if pooling == 'avg' else nn.AdaptiveMaxPool3d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        # activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear)


class Selayer(nn.Module):

    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class BottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, kernel=(3, 3, 3), padding=(1, 1, 1), downsample=None):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes * 2)

        self.conv2 = nn.Conv3d(planes * 2, planes * 2, kernel_size=kernel, stride=stride,
                               padding=padding, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(planes * 2)

        self.conv3 = nn.Conv3d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.selayer = Selayer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNeXt(nn.Module):

    def __init__(self, block, layers, int_channel=1, cardinality=32,depth=5):
        super(SEResNeXt, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64

        self.conv1 = nn.Conv3d(int_channel, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))
        if depth == 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, kernel=3, padding=1)
        elif depth == 5:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, kernel=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] *  m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel=(3, 3, 3), padding=(1, 1, 1)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, kernel, padding, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, kernel=kernel, padding=padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        feature.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feature.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)

        return feature

def se_resnet50(**kwargs):
    """Constructs a SE-ResNet-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet101(**kwargs):
    """Constructs a SE-ResNet-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet152(**kwargs):
    """Constructs a SE-ResNet-152 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def se_resnext50(**kwargs):
    """Constructs a SE-ResNeXt-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model


def se_resnext101(**kwargs):
    """Constructs a SE-ResNeXt-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(BottleneckX, [3, 4, 23, 3], **kwargs)
    return model


def se_resnext152(**kwargs):
    """Constructs a SE-ResNeXt-152 model.
    Args:
        num_classes = 1000 (default)
    """
    model = SEResNeXt(BottleneckX, [3, 8, 36, 3], **kwargs)
    return model




class Unet(nn.Module):
    def __init__(self,
                 int_channel=1, classes=1, aux_params=None, encoder_name='se_resnet50'):
        super(Unet, self).__init__()
        if encoder_name == 'se_resnet50':
            self.encoder = se_resnet50(int_channel=int_channel)
            self.ecoder_channels = [1, 64, 256, 512, 1024, 2048]
            self.decoder_channels = [256, 128, 64, 32, 16]
        elif encoder_name == 'resnext50':
            self.encoder = se_resnext50(int_channel=int_channel)
            self.ecoder_channels = [1, 64, 256, 512, 1024, 2048]
            self.decoder_channels = [256, 128, 64, 32, 16]

        self.decoder = UnetDecoder(self.ecoder_channels, self.decoder_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1], out_channels=classes,
            kernel_size=3)
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.ecoder_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks


class Enocode(nn.Module):
    def __init__(self,
                 int_channel=1, classes=1, aux_params=None, encoder_name='se_resnet50'):
        super(Enocode, self).__init__()
        if encoder_name == 'se_resnet50':
            self.encoder = se_resnet50(int_channel=int_channel)
            self.ecoder_channels = [1, 64, 256, 512, 1024, 2048]
            self.decoder_channels = [256, 128, 64, 32, 16]
        elif encoder_name == 'resnext50':
            self.encoder = se_resnext50(int_channel=int_channel)
            self.ecoder_channels = [1, 64, 256, 512, 1024, 2048]
            self.decoder_channels = [256, 128, 64, 32, 16]

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.ecoder_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)


        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return labels
