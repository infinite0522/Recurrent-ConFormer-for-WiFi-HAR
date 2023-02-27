from torch import nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Recurrent_Res_block(nn.Module):
    '''
    Recurrent convolution block for RU-Net and R2U-Net
    Args:
        ch_out : number of outut channels
        t: the number of recurrent convolution block to be used
    Returns:
        feature map of the given input
    '''

    def __init__(self, planes, t=2):
        super(Recurrent_Res_block, self).__init__()
        self.t = t
        self.ch_out = planes
        self.conv1 = conv3x3(planes, planes)
        self.conv2 = conv3x3(planes, planes)
        self.conv01 = conv3x3(planes, planes)
        self.conv02 = conv3x3(planes, planes)
        self.bn = nn.ModuleList([nn.BatchNorm1d(planes) for i in range(2 * t)])
        self.bn0 = nn.ModuleList([nn.BatchNorm1d(planes) for i in range(2 * t)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rx = x
        for i in range(self.t):
            x1 = self.conv1(x)
            x1 = self.bn[2 * i](x1)
            x1 = self.relu(x1)
            x1 = self.conv2(x1)
            x = self.bn[2 * i + 1](x1)
            if i != 0:
                x0 = self.conv01(rx)
                x0 = self.bn0[2 * i](x0)
                x0 = self.relu(x0)
                x0 = self.conv02(x0)
                x0 = self.bn0[2 * i + 1](x0)
                x = x + x0
        x = self.relu(rx + x)

        return x
