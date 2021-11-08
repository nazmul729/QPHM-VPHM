'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.quaternionconv.quaternion_layers import QuaternionConv
from lib.models.quaternionconv.quaternion_QLinearlayers import QLinear

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = QuaternionConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = QuaternionConv(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuaternionConv(planes, self.expansion*planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))         #print("Input: ", x.shape, "Out: ",out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    expansion = 4
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 120#112
        self.num_classes = num_classes
        self.conv1 = QuaternionConv(4, 120, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(120) 
        self.layer1 = self._make_layer(block, 120, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 240, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 480, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 960, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.linear = nn.Linear(896*block.expansion, num_classes)#
        self.linear = QLinear(5, 960*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def Layer_Norm(self, x, num_planes=128):
        mean = x.sum(axis = 0)/(x.shape[0])
        std = ((((x - mean)**2).sum()/(x.shape[0]))+0.00001).sqrt()
        return (x - mean)/std

    def forward(self, x):
        #print("Before stem: ",x.shape) um = self.num_classes  #print(num)  #10
        out = F.relu(self.bn1(self.conv1(x)))   #out = self.maxpool(out)
        out = self.layer1(out)    #  print("Layer1: ",out.shape) 
        out = self.layer2(out)    #  print("Layer2: ",out.shape)
        out = self.layer3(out)    #  print("Layer3: ",out.shape)
        out = self.layer4(out)    #  print("Layer4: ",out.shape)
        
        out = self.avgpool(out)              #print(out.shape) [500, 57344]
        out = torch.flatten(out, 1)
        out = self.linear(out)                  #print(out.shape)
        
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet26(num_classes):
    return ResNet(BasicBlock, [2, 4, 4, 2], num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == '__main__':
    net = ResNet18(1000)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    y = net(torch.randn(128, 3, 256, 256))
    print(y.size())
