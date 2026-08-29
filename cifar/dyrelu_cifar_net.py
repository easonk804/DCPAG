import torch
import torch.nn as nn

from .modules import (
    DynamicReLUV1,
    active_channel_count,
    collect_gate_l1,
    validate_ratio,
)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # GAP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # fc-relu-fc-sigmoid
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # size-->(batch_size, C, H, W)
        b, c, _, _ = x.size()
        # GAP + view
        y = self.avg_pool(x).view(b, c)
        # fc + view
        y = self.fc(y).view(b, c, 1, 1)
        # 将y的形状弄成和a一样，比如说a的特征图大小(2,2),b的大小本来是(1,1)，那么就把那个值复制四次然后把大小弄成和a的一样
        return x * y.expand_as(x)

class DynamicConvBnReLU(nn.Module):
    '''带有Dynamic_relu的卷积组件'''
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=1.0, K=2, reduction=8):
        super(DynamicConvBnReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ratio = validate_ratio(ratio)
        self.K = K
        self.reduction = reduction
        self.last_gate_l1 = None

        # conv-bn-dynamic_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dynamic_relu = DynamicReLUV1(in_channels, out_channels, K, reduction)

    def forward(self, input):
        # conv-bn-dynamic_relu
        x = self.conv(input)
        x = self.bn(x)
        a, b = self.dynamic_relu(input)

        # 不激活的通道个数
        inactive_channels = self.out_channels - active_channel_count(
            self.out_channels, self.ratio
        )
        # gates, _ = torch.max(a, dim=0)
        # 得到每个样本最大值
        gates, _ = torch.max(torch.abs(a), dim=0)
        self.last_gate_l1 = a.abs().sum() / input.size(0)

        # 得到值大的通道索引
        if inactive_channels:
            inactive_idx = (-gates).topk(inactive_channels, dim=1).indices
            inactive_idx = inactive_idx.unsqueeze(0).expand(self.K, -1, -1, -1, -1)
            # Avoid in-place mutation of tensors needed by autograd.
            a = a.scatter(2, inactive_idx, 0)
            b = b.scatter(2, inactive_idx, 0)

        # dynamic_relu的形式：max{a*x+b}
        x = a*x + b
        x, _ = torch.max(x, dim=0)

        return x


class DyReLUCifarNet(nn.Module):
    '''带有Dynamic_relu的CifarNet'''
    def __init__(self, ratio=1.0, num_classes=10):
        super(DyReLUCifarNet, self).__init__()
        validate_ratio(ratio)
        self.gconv0 = DynamicConvBnReLU(3, 64, padding=0, ratio=ratio)
        self.gconv1 = DynamicConvBnReLU(64, 64, ratio=ratio)
        self.gconv2 = DynamicConvBnReLU(64, 128, stride=2, ratio=ratio)
        self.gconv3 = DynamicConvBnReLU(128, 128, ratio=ratio)
        self.drop3 = nn.Dropout2d()
        self.gconv4 = DynamicConvBnReLU(128, 128, ratio=ratio)
        self.gconv5 = DynamicConvBnReLU(128, 192, stride=2, ratio=ratio)
        self.gconv6 = DynamicConvBnReLU(192, 192, ratio=ratio)
        self.drop6 = nn.Dropout2d()
        self.gconv7 = DynamicConvBnReLU(192, 192,ratio=ratio)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.gconv0(x)
        x = self.gconv1(x)
        x = self.gconv2(x)
        x = self.gconv3(x)
        x = self.drop3(x)
        x = self.gconv4(x)
        x = self.gconv5(x)
        x = self.gconv6(x)
        x = self.drop6(x)
        x = self.gconv7(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        gate_l1_norm = collect_gate_l1(self, x)

        return  x, gate_l1_norm

if __name__ == '__main__':
    model = DyReLUCifarNet(ratio=0.5)
    x = torch.rand(2, 3, 32, 32)
    out, l1_norm = model(x)
    print(out.size(), l1_norm)
    params = sum(param.nelement() for param in model.parameters())
    print('total params:{0}'.format(params))
