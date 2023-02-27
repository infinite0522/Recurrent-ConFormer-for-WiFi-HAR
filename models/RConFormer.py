import numpy as np
import torch
from torch import nn
from torch.nn.utils.weight_norm import WeightNorm

#from apl import *
from .RecurrentRes import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
    def __init__(self, n_way: int = 6, n_feature: int = 128, length: int = 192,
                 n_head: int = 8, n_encoder_layers: int = 1, t: int = 4,
                 dim_projection: int = 128, dim_feedforward: int = 256,
                 loss_type: str = 'dist'):
        super(TransformerModel, self).__init__()

        self.n_feature = n_feature
        self.n_head = n_head
        self.dim_projection = dim_projection
        self.d_model = dim_projection
        self.dim_feedforward = dim_feedforward
        self.n_encoder_layers = n_encoder_layers
        self.t = t
        self.loss_type = loss_type  # 'softmax' #'dist'

        # Patch encoder
        self.patch_encoder = nn.Linear(self.n_feature, self.dim_projection)
        # 直接transformer：193:timestamp+1；一层CNN后：49（48+1）；
        self.len = int(length / 4) + 1
        self.pos_encoder = nn.Parameter(torch.randn(self.len, self.dim_projection) * 1e-1)

        # Encoder Layer
        """encoder_layer0, encoder_norm0, encoder0为冗余代码，但删掉后会因随机初始化不同导致训练结果与论文略有不同"""
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                                   dim_feedforward=dim_feedforward)
        encoder_layer0 = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                                    dim_feedforward=dim_feedforward)
        encoder_norm = nn.LayerNorm(self.d_model)
        encoder_norm0 = nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers, encoder_norm)
        self.encoder0 = nn.TransformerEncoder(encoder_layer0, n_encoder_layers, encoder_norm0)

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.d_model, n_way)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':
            self.classifier = distLinear(self.d_model, n_way)

    def forward(self, src):
        src = src / torch.max(src)
        batch_sz = src.shape[1]
        src = self.patch_encoder(src)  # dim_projection=256
        src += torch.unsqueeze(self.pos_encoder, 1).repeat(1, batch_sz, 1)

        """recurrent tranformer"""
        memory = src
        for i in range(self.t):
            if i != 0:
                memory = memory + src
            memory = self.encoder(memory)

        memory = memory[-1]  # cls token
        scores = self.classifier(memory)

        return scores


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)

        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)

        return scores


class R_ConFormer(nn.Module):
    def __init__(self, n_way: int = 6, input_shape: int = [52, 192],
                 n_head: int = 8, n_cnn_layers: int = 1, n_encoder_layers: int = 1, t_encoder: int = 4,
                 dim_projection: int = 128, dim_feedforward: int = 256,
                 loss_type: str = 'dist'):
        super(R_ConFormer, self).__init__()
        self.inplanes = dim_projection
        self.conv1 = nn.Conv1d(input_shape[0], dim_projection, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(dim_projection)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)     #conv1(kernel=7)+maxpool 后timestamp变成48

        self.rcnn = Recurrent_Res_block(dim_projection, t=n_cnn_layers)
        self.transformer = TransformerModel(n_way=n_way, n_feature=dim_projection, length=input_shape[1], n_head=n_head,
                                            n_encoder_layers=n_encoder_layers, t=t_encoder, dim_projection=dim_projection,
                                            dim_feedforward=dim_feedforward)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.rcnn(x)

        cls = torch.zeros((x.shape[0], x.shape[1], 1)).to(device)
        x = torch.cat((x, cls), dim=2)
        x = torch.permute(x, (2, 0, 1))
        x = self.transformer(x)

        return x
