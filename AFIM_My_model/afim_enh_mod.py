
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import afim_mod_wt
sys.path.append('G:/Sharda_Code/V6/AFIM/AFIM_My_model/')
from afim_mod_layer import AFIMDeepConv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten

class AFIMInitBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, n=4):
        super(AFIMInitBlock, self).__init__()
        self.conv1 = AFIMDeepConv(n,
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AFIMDeepConv(n, planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                AFIMDeepConv(n, in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride,),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, n=4):
        super(Bottleneck, self).__init__()
        self.conv1 = AFIMDeepConv(n, in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AFIMDeepConv(n, planes, planes, kernel_size=3,
                               stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = AFIMDeepConv(n, planes, self.expansion * planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                AFIMDeepConv(n, in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class AFIMDeepNet(nn.Module):
    def __init__(self, block, num_blocks, channels=4, n=4, num_classes=10, before_gap_output=False, gap_output=False, visualize=False):
        super(AFIMDeepNet, self).__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.n = n
        self.before_gap_out = before_gap_output
        self.gap_output = gap_output
        self.visualize = visualize
        self.conv1 = AFIMDeepConv(n, channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, n=n)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, n=n)
        self.layer5 = None
        self.layer6 = None
        
        if not before_gap_output and not gap_output:
            self.linear = nn.Linear(512*block.expansion, num_classes)
        
    def add_top_blocks(self, num_classes=1):
        
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2, n=self.n)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2, n=self.n)
        
        if not self.before_gap_out and not self.gap_output:
            self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out4 = self.layer4(out)
        
        if self.before_gap_out:
            return out4
        
        if self.layer5:
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)
        n, c, _, _ = out6.size()
        out = out6.view(n, c, -1).mean(-1)
        
        if self.gap_output:
            return out

        out = self.linear(out)

        if self.visualize:
            
            return out, out4, out6
        return out

class AFIMAttentenc(nn.Module):
    

    def __init__(self, channels, n):
        super(AFIMAttentenc, self).__init__()
        self.in_planes = 64

        self.conv1 = AFIMDeepConv(n, channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(AFIMInitBlock, 64, 2, stride=1, n=n)
        self.layer2 = self._make_layer(AFIMInitBlock, 128, 2, stride=2, n=n)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class SharedBottleneck(nn.Module):
    def __init__(self, n, in_planes):
        super(SharedBottleneck, self).__init__()
        self.in_planes = in_planes
        
        self.layer3 = self._make_layer(AFIMInitBlock, 256, 2, stride=2, n=n)
        self.layer4 = self._make_layer(AFIMInitBlock, 512, 2, stride=2, n=n)
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        n, c, _, _ = out.size()
        out = out.view(n, c, -1).mean(-1)
        return out

class AFIMClassModel(nn.Module):
    

    def __init__(self, n, num_classes, in_planes=512, visualize=False):
        super(AFIMClassModel, self).__init__()
        self.in_planes = in_planes
        self.visualize = visualize

        
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2, n=n)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer5(x)
        feature_maps = self.layer6(out)

        n, c, _, _ = feature_maps.size()
        out = feature_maps.view(n, c, -1).mean(-1)
        out = self.linear(out)

        if self.visualize:
            return out, feature_maps

        return out

class AFIMDeepEncModNet3(nn.Module):
    def __init__(self, n, hyper_param3=True, num_classes=1, weights=None):
        super(AFIMDeepEncModNet3, self).__init__()
        
        self.hyper_param3 = hyper_param3
        
        self.encoder_sx = AFIMAttentenc(channels=2, n=2)
        self.encoder_dx = AFIMAttentenc(channels=2, n=2)
        
        self.shared_resnet = SharedBottleneck(n, in_planes=128 if hyper_param3 else 256)
        
        if weights:
            afim_mod_wt(self.encoder_sx, weights)
            afim_mod_wt(self.encoder_dx, weights)
        
        self.classifier_sx = nn.Linear(1024, num_classes)
        self.classifier_dx = nn.Linear(1024, num_classes)

    def forward(self, x):
        x_sx, x_dx = x
        
        # Apply AFIMAttentenc
        out_sx = self.encoder_sx(x_sx)
        out_dx = self.encoder_dx(x_dx)
        
        # Shared layers
        if self.hyper_param3:
            out_sx = self.shared_resnet(out_sx)
            out_dx = self.shared_resnet(out_dx)
            
            out_sx = self.classifier_sx(out_sx)
            out_dx = self.classifier_dx(out_dx)
            
        else: # Concat version  
            out = torch.cat([out_sx, out_dx], dim=1)
            out = self.shared_resnet(out)
            out_sx = self.classifier_sx(out)
            out_dx = self.classifier_dx(out)
        
        out = torch.cat([out_sx, out_dx], dim=0)
        return out

class AFIMDeepEncModNet4(nn.Module):
    

    def __init__(self, n=2, num_classes=1, weights=None, hyper_param2=True, visualize=False):
        super(AFIMDeepEncModNet4, self).__init__()
        self.visualize = visualize
        self.AFIMDeepNetmod1 = AFIMDeepNetmod1(n=2, num_classes=num_classes, channels=2, before_gap_output=True)
        
        if weights:
            print("Loading model_weights for AFIMDeepNetmod1 from ", weights)
            afim_mod_wt(self.AFIMDeepNetmod1, weights)
        
        self.classifier_sx = AFIMClassModel(n, num_classes, visualize=visualize)
        self.classifier_dx = AFIMClassModel(n, num_classes, visualize=visualize)

        if not hyper_param2 and weights:
            print("Loading model_weights for classifiers from ", weights)
            afim_mod_wt(self.classifier_sx, weights)
            afim_mod_wt(self.classifier_dx, weights)
    
    def forward(self, x):
        x_sx, x_dx = x
        
        
        out_enc_sx = self.AFIMDeepNetmod1(x_sx)
        out_enc_dx = self.AFIMDeepNetmod1(x_dx)
        
        if self.visualize:
            out_sx, act_sx = self.classifier_sx(out_enc_sx)
            out_dx, act_dx = self.classifier_dx(out_enc_dx)
        else:
            
            out_sx = self.classifier_sx(out_enc_sx)
            out_dx = self.classifier_dx(out_enc_dx)
        
        out = torch.cat([out_sx, out_dx], dim=0)

        if self.visualize:
            return out, out_enc_sx, out_enc_dx, act_sx, act_dx

        return out

def AFIMDeepNetmod1(channels=4, n=4, num_classes=10, before_gap_output=False, gap_output=False, visualize=False):
    return AFIMDeepNet(AFIMInitBlock, 
                    [2, 2, 2, 2], 
                    channels=channels, 
                    n=n, 
                    num_classes=num_classes, 
                    before_gap_output=before_gap_output, 
                    gap_output=gap_output,
                    visualize=visualize)

def AFIMDeepNetmod2(channels=4, n=4, num_classes=10):
    return AFIMDeepNet(Bottleneck, [3, 4, 6, 3], channels=channels, n=n, num_classes=num_classes)
def afim_deepnet():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 1)))
    model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
    return model
