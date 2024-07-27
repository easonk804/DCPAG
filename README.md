# DCPAG-
作者Shunqiang Liu于2021年4月开发了基于动态掩码ReLU的模型减枝管道，并将其应用于解决模型减枝后的性能问题。
https://link.springer.com/article/10.1007/s10489-022-03383-w

## DyReLUCifarNet
这个文件实现了一个名为DyReLUCifarNet的创新网络结构,主要有以下几个创新点:

- 动态ReLU激活函数: 使用DynamicReLUV1替代传统ReLU,可以自适应地调整激活函数的形状。

- 通道剪枝: 通过ratio参数控制每层激活的通道数量,实现了动态的通道剪枝。

- 门控机制: 使用gates来选择最重要的通道,不重要的通道会被置零。

- L1正则化: 计算门函数的L1范数,可用于正则化训练过程。

- 模块化设计: 使用DynamicConvBnReLU模块封装了卷积、批归一化和动态ReLU,便于构建网络。

- 灵活性: 通过ratio参数可以灵活调整网络的计算复杂度和参数量。

这些创新点使得DyReLUCifarNet能够在保持性能的同时,实现动态的模型压缩和加速。

## DyReLUResNet
这个文件实现了一个创新的ResNet变体，主要有以下几个创新点：

- 动态ReLU激活函数：使用DynamicReLU和DynamicGatedReLU替代了传统的ReLU激活函数。这些动态激活函数可以根据输入自适应地调整其参数，提高网络的表达能力。

- 门控机制：BasicGatedBlock引入了门控机制，可以动态调节特征的重要性。这有助于网络更好地关注重要特征，忽略不重要的信息。

- L1范数正则化：通过计算门控值的L1范数(cal_gate_l1_norm函数)，实现了对门控机制的正则化，有助于控制模型复杂度和防止过拟合。

- 灵活的网络配置：提供了不同深度(18/34/50/101)的ResNet变体，并允许通过ratio参数调整动态激活函数的行为。

- CIFAR数据集适配：网络结构针对CIFAR数据集进行了调整，如第一个卷积层使用3x3卷积核而非7x7，去掉了最初的最大池化层。

这些创新点使得网络能够更好地适应不同的任务和数据集，同时提高了模型的表达能力和泛化性能。

