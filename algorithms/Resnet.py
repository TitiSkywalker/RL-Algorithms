import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.is_equal = in_planes==planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = (not self.is_equal) and nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.add(x if self.is_equal else self.shortcut(x), out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channels,input_channels=3,output_size=10,is_ppo=False,need_softmax=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[2])
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(channels[2], output_size)
        self.need_softmax = need_softmax
        if need_softmax:
            self.softmax = nn.Softmax(dim=-1)
        self.is_ppo = is_ppo
        self.output_size = output_size
        if is_ppo:
            self.p = nn.Sequential(
                nn.Linear(channels[2],channels[2]),
                nn.ReLU(),
                nn.Linear(channels[2],output_size),
                nn.Softmax(dim=-1)
            )
            self.q = nn.Sequential(
                nn.Linear(channels[2],channels[2]),
                nn.ReLU(),
                nn.Linear(channels[2],1)
            )
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if self.is_ppo:
            return self.p(out),self.q(out)
        else:
            if self.need_softmax:
                return self.softmax(self.linear(out))
            else:
                return self.linear(out)

def WideResNet(input_channels=3,output_size=10,is_ppo=False,need_softmax=False):
    return ResNet(BasicBlock, [9, 9, 9], [128,256,512],input_channels=input_channels,output_size=output_size,is_ppo=is_ppo,need_softmax=need_softmax)

def WideResNet_small(input_channels=3,output_size=10,is_ppo=False,need_softmax=False):
    return ResNet(BasicBlock, [9, 9, 9], [32,64,128],input_channels=input_channels,output_size=output_size,is_ppo=is_ppo,need_softmax=need_softmax)

def WideResNet_small_small(input_channels=3,output_size=10,is_ppo=False,need_softmax=False):
    return ResNet(BasicBlock, [3,3,3], [32,64,128],input_channels=input_channels,output_size=output_size,is_ppo=is_ppo,need_softmax=need_softmax)

