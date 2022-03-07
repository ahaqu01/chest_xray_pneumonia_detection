# import torch
# import torchvision
# from unet import UNet
# from resnet import ResNet
# model =ResNet()#自己定义的网络模型
# model.load_state_dict(torch.load("model.pth"))#保存的训练模型
# model.eval()#切换到eval（）
# example = torch.rand(1, 3, 320, 480)#生成一个随机输入维度的输入
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("model.pt")

# 针对有网络模型，但还没有训练保存 .pth 文件的情况
import torch.onnx
from torch.autograd import Variable
from torchvision.models import resnet18  # 以 resnet18 为例

myNet = resnet18()  # 实例化 resnet18
x = torch.randn(16, 3, 40, 40)  # 随机生成一个输入
modelData = "./demo.pth"  # 定义模型数据保存的路径
# modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的
torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
