import torch
import torch.nn as nn

from .modules import active_channel_count, collect_gate_l1, validate_ratio

class GatedBN(nn.Module):
    def __init__(self, inplanes, outplanes, ratio):
        super(GatedBN, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.ratio = validate_ratio(ratio)
        self.bn = nn.BatchNorm2d(outplanes, affine=False)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.gate = nn.Linear(inplanes, outplanes)
        self.beta = nn.Parameter(torch.zeros(outplanes))
        self.relu = nn.ReLU(inplace=True)
        self.last_gate_l1 = None
        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.gate.bias, 1.0)
        # nn.init.kaiming_normal_(self.gate.weight)

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        x = self.global_pool(input1)
        x = x.view(batch_size, -1)
        gates = self.relu(self.gate(x))
        self.last_gate_l1 = gates.abs().sum() / batch_size

        beta = self.beta.repeat(batch_size, 1)

        if self.ratio < 1:
            inactive_channels = self.outplanes - active_channel_count(
                self.outplanes, self.ratio
            )
            inactive_idx = (-gates).topk(inactive_channels, 1)[1]
            gates = gates.scatter(1, inactive_idx, 0)  # set inactive channels as zeros
            beta = beta.scatter(1, inactive_idx, 0)

        x = self.bn(input2)
        x = gates.unsqueeze(2).unsqueeze(3) * x
        x = x + beta.unsqueeze(2).unsqueeze(3)

        return x

class GatedConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1, ratio=1.0):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = GatedBN(inplanes, outplanes, ratio)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(input, x)
        x = self.relu(x)

        return x

class FBSCifarNet(nn.Module):
    def __init__(self, ratio=1.0, num_classes=10):
        super(FBSCifarNet, self).__init__()
        ratio = validate_ratio(ratio)
        self.gconv0 = GatedConv(3, 64, padding=0, ratio=ratio)
        self.gconv1 = GatedConv(64, 64, ratio=ratio)
        self.gconv2 = GatedConv(64, 128, stride=2, ratio=ratio)
        self.gconv3 = GatedConv(128, 128, ratio=ratio)
        self.drop3 = nn.Dropout2d()
        self.gconv4 = GatedConv(128, 128, ratio=ratio)
        self.gconv5 = GatedConv(128, 192, stride=2, ratio=ratio)
        self.gconv6 = GatedConv(192, 192, ratio=ratio)
        self.drop6 = nn.Dropout2d()
        self.gconv7 = GatedConv(192, 192,ratio=ratio)
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
    model = FBSCifarNet(True)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    x = torch.rand((2,3,32,32))
    out, gate_l1_norm = model.forward(x)
    print('out size:{0}, gate_l1_norm:{1}'.format(out.size(), gate_l1_norm))
