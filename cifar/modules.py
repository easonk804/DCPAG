import torch
import torch.nn as nn


def validate_ratio(ratio):
    """Validate and normalize the fraction of active channels."""
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    return ratio


def active_channel_count(channels, ratio):
    """Return a valid number of active channels for a pruning ratio."""
    return max(1, round(channels * validate_ratio(ratio)))


def collect_gate_l1(module, reference):
    """Sum per-layer gate penalties recorded during the current forward pass."""
    penalties = [
        child.last_gate_l1
        for child in module.modules()
        if torch.is_tensor(getattr(child, "last_gate_l1", None))
    ]
    if not penalties:
        return reference.new_zeros(())
    return torch.stack(penalties).sum()

class TorchGraph(nn.Module):
    '''Torch图'''
    def __init__(self):
        super(TorchGraph, self).__init__()
        # 创建一个图字典
        self._graph = {}

    def add_tensor_list(self, name):
        # 向图字典里追加张量列表
        self._graph[name] = []

    def append_tensor(self, name, val):
        # 向图字典的张量列表里添加张量
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        # 清空张量列表
        self._graph[name].clear()

    def get_tensor_list(self, name):
        # 获得张量列表
        return self._graph[name]

    def set_global_var(self, name, val):
        # 设置全局变量
        self._graph[name] = val

    def get_global_var(self, name):
        # 获得全局变量
        return self._graph[name]


class HardSigmoid(nn.Module):
    """Piecewise-linear sigmoid used by Dynamic ReLU."""
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu(x+3)/6

class DynamicReLU(nn.Module):
    '''动态ReLU'''
    def __init__(self, inplanes, outplanes, K=2, reduction=8):
        super(DynamicReLU, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.K = K
        self.reduction = reduction
        self.middle_planes = max(inplanes // reduction, 2)

        # channel attention
        # 全局平均池化(将特征图大小弄成1x1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fc1
        self.fc1 = nn.Linear(inplanes, self.middle_planes)
        # fc2
        self.fc2 = nn.Linear(self.middle_planes, 2*K*outplanes)
        # relu
        self.relu = nn.ReLU(inplace=True)
        # hard sigmoid
        self.sigmoid = HardSigmoid(inplace=True)

        # Fixed base coefficients belong in buffers, not the optimizer.
        alpha = torch.zeros(K, 1)
        alpha[0][0] = 1.0
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", torch.zeros(K, 1))

        # 控制超参数a和b的scaler
        self.lambda_a = 1.0
        self.lambda_b = 0.5

    def forward(self, input):
        # batch_size
        batch_size = input.size(0)
        # GAP
        x = self.global_pool(input)
        # Flatten
        x = x.view(batch_size, -1)
        # fc1
        x = self.fc1(x)
        # relu
        x = self.relu(x)
        # fc2
        x = self.fc2(x)
        # 2*sigmoid-1
        x = 2 * self.sigmoid(x) - 1.0
        # Preserve sample boundaries. The previous reshape mixed batch and
        # channel values whenever batch_size > 1.
        x = x.reshape(batch_size, 2, self.K, self.outplanes)
        x = x.permute(1, 2, 0, 3).contiguous()
        # delta_a和delta_b
        delta_a = x[0]
        delta_b = x[1]
        # 更新a和b
        a = self.alpha[:, None, :] + self.lambda_a * delta_a
        b = self.beta[:, None, :] + self.lambda_b * delta_b
        # Flatten(Kernel, m, C, H, W)
        a = a.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)

        return a, b


class DynamicReLUV1(nn.Module):
    '''动态ReLUV1'''
    def __init__(self, inplanes, outplanes, K=2, reduction=8):
        super(DynamicReLUV1, self).__init__()
        # inplanes
        self.inplanes = inplanes
        # outplanes
        self.outplanes = outplanes
        # Kernel
        self.K = K
        # reduction
        self.reduction = reduction
        # middle_planes
        self.middle_planes = max(inplanes // reduction, 2)

        # channel attention
        # GAP
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fc
        self.fc = nn.Linear(inplanes, K*outplanes)
        # relu
        self.relu = nn.ReLU(inplace=True)
        # sigmoid
        self.sigmoid = HardSigmoid(inplace=True)

        # alpha
        alpha = torch.zeros(K, 1)
        alpha[0][0] = 1.0
        self.register_buffer("alpha", alpha)
        self.lambda_a = 1.0

        # beta
        self.b = torch.nn.Parameter(torch.zeros(K*outplanes))

    def forward(self, input):
        # batch_size
        batch_size = input.size(0)
        # GAP
        x = self.global_pool(input)
        # Flatten
        x = x.view(batch_size, -1)
        # fc
        x = self.fc(x)
        # sigmoid
        x = 2*self.sigmoid(x) - 1.0

        # (batch, K * channels) -> (K, batch, channels)
        delta_a = x.reshape(batch_size, self.K, self.outplanes)
        delta_a = delta_a.permute(1, 0, 2).contiguous()

        # 更新a和b
        a = self.alpha[:, None, :] + self.lambda_a * delta_a
        b = self.b.reshape(self.K, self.outplanes)[:, None, :]
        b = b.expand(-1, batch_size, -1)

        # Flatten(K, m, C, H, W)
        a = a.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)

        return a, b


class DynamicGatedReLU(nn.Module):
    '''动态门ReLU'''
    def __init__(self, inplanes, outplanes, ratio, K=2, reduction=8):
        super(DynamicGatedReLU, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.ratio = validate_ratio(ratio)
        self.K = K
        self.reduction = reduction
        self.last_gate_l1 = None

        self.dyrelu = DynamicReLU(inplanes, outplanes, K, reduction)

    def forward(self, input1, input2):
        # 获得超参数a和b
        a, b = self.dyrelu(input1)
        # 得到不激活的通道数
        inactive_channels = self.outplanes - active_channel_count(
            self.outplanes, self.ratio
        )
        # 留下值较大的那些Kernel，返回的是(value, indices)
        gates, _ = torch.max(a, dim=0)
        self.last_gate_l1 = gates.abs().sum() / input1.size(0)
        # 对于不同的输入样本找到c个值较大的通道，返回的是(value, indices)，这里我们要的是indices
        if inactive_channels:
            inactive_idx = (-gates).topk(inactive_channels, dim=1).indices
            inactive_idx = inactive_idx.unsqueeze(0).expand(self.K, -1, -1, -1, -1)
            a = a.scatter(2, inactive_idx, 0)
            b = b.scatter(2, inactive_idx, 0)

        # 动态ReLU函数的表达式：max(ax+b)
        x = a * input2 + b
        x, _ = torch.max(x, dim=0)

        return x, gates
