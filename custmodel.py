import torch
import torch.nn as nn

class custom_resnet(nn.Module):
  def __init__(self, resnet):
    super(custom_resnet, self).__init__()
    self.resnet = resnet
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(4096,1024),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(1024, 1024),
      nn.ReLU(inplace=True),
      nn.Linear(1024, 5)
    )

  def forward(self, x1):
    x1 = self.resnet.conv1(x1)
    x1 = self.resnet.bn1(x1)
    x1 = self.resnet.relu(x1)
    x1 = self.resnet.maxpool(x1)

    x1 = self.resnet.layer1(x1)
    x1 = self.resnet.layer2(x1)
    x1 = self.resnet.layer3(x1)
    x1 = self.resnet.layer4(x1)
        
    x1 = self.resnet.avgpool(x1)
    x1 = x1.view(x1.size(0), -1)
        

    p = self.classifier(x1)
    return p

