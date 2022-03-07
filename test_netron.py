import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch = 2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.mp = nn.MaxPool2d(2)
        self.mp1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2560, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp1(self.conv3(x)))
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

#torch.save(model, './model_para.pth')
torch.save(model.state_dict(), './model_para.pth')
# state = {
#     'state':model.state_dict(),
#     'epoch':epoch
# }
# torch.save(state, './model_para.pth')

# 针对有网络模型，但还没有训练保存 .pth 文件的情况
import torch.onnx
from torch.autograd import Variable
from torchvision.models import resnet18  # 以 resnet18 为例

# myNet = resnet18()  # 实例化 resnet18
# myNet = torch.load("model_para.pth" )
myNet = Net()
x = torch.randn(10, 1, 5, 5)  # 随机生成一个输入
modelData = "./model_para_1.pth"  # 定义模型数据保存的路径
# modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的
torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
