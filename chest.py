# 1.加载库
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models, utils
import time
import os
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboader.writer import SummaryWriter
import matplotlib.pyplot as plt
import torchvision
import copy

# 2.定义方法显示图片
def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose(1, 2 ,0) # 转置
    mean = np.array([0.485, 0.456, 0.406]) # 均值
    std = np.array([0.229, 0.224, 0.225]) # 标准差
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()

# 8.更改池化层
class AdaptiveConcatPool2d(nn.Module): #构造一个自适应的池化层
    def __init__(self, size=None):
        super().__init__()
        size = size or (1, 1) # 池化层的卷积核大小，默认值为（1,1）
        self.pool_one = nn.AdaptiveAvgPool2d(size) #池化层1
        self.pool_two = nn.AdaptiveAvgPool2d(size) #池化层2

    def forward(self, x):
        return torch.cat(self.pool_one(x), self.pool_two(x), 1) #连接两个池化层

# 7. 迁移学习：拿到一个成熟的模型，进行模型微调
def get_model():
    #model_pre = models.resnet50(pretrained=True) # 获取预训练模型
    model_pre = models.yolov5(pretrained=True)
    # models_pre = models.resnet101(pretrained=True)
    # 冻结预训练模型中所有的参数
    for param in model_pre.parameters(): # 获取模型的所有参数
        param.requires_grad = False # 将所有参数冻结起来
    # 微调模型：替换ResNet最后的两层网络，返回一个新的模型
    model_pre.avgpool = nn.AdaptiveConcatPool2d() # 池化层替换。 model_pred.avgpool:指modle_pred的池化层。
    model_pre.fc = nn.Sequential( # 全连接层替换
        nn.Flatten(), #所有维度拉平
        nn.BatchNorm1d(4096), # 正则化处理。256x6x6=9216，最后要变成4096，扔掉了一些参数，4096表示卷积的大小，卷积的数量是4096。具体不知道
        nn.Dropout(0.5), # 防止过拟合，丢掉一些神经元，0.5是最好的，网络的结构是最丰富的
        nn.Linear(4096, 512), # 线性层的处理
        nn.ReLU(), # 激活函数,激活层
        nn.BatchNorm1d(512), # 正则化处理
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1) # 损失函数，计算分类后，每个数字的概率值，并返回概率最高的数字。
    )

    return model_pre

# 9.定义一个训练函数
# criterion:损失函数，model训练的模型， device设备， train_loader训练的数据，optimizer优化器， epoch训练的轮次,
# writer把训练过程中的一些值（准确度，损失值等等）写入到日志当中，后期可以通过pytorch的tensorboardx库来绘制这个图形，可以看到训练的整个图形
def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train() # 训练过程中，会进行一个梯度的计算
    total_loss = 0.0 # 总损失值初始化 为0
    # 循环读取训练数据集，更新模型参数
    for batch_id, (data, target) in enumerate(train_loader): # batch_id指的是第几个批次，每一批有batch_size张图训练
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 梯度初始化为0
        output = model(data) # 训练后的输出
        loss = criterion(output, target).item() #计算损失
        loss.backward() #反向传播
        optimizer.step() #参数更新
        total_loss += loss.item() #累计训练损失

    writer.add_scalar("Train Loss ", total_loss/len(train_loader) , epoch)
    writer.flush() #刷新

    return total_loss / len(train_loader) #返回平均损失

# 10.定义测试函数
def test(model,device, test_loader,criterion, epoch, writer):
    model.eval() #测试要声明一下不需要训练函数，不需要Dropout
    # 正确与损失
    correct = 0.0
    test_loss = 0.0
    # 循环读取数据
    with torch.no_grad(): # 不用计算梯度，也不用进行反向传播
        for data, target in test_loader:
            # 部署到device上去
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += criterion(output, target).item() #计算损失
            # 获取预测结果中每行数据概率最大的下标
            pred = torch.max(output, dim=1)
            # 累计正确的值或数目
            correct += torch.sum(pred == target).item()
        # 平均损失
        test_loss /= len(test_loader.dataset)
        # 正确率
        accuracy = correct / len(test_loader.dataset)
        # 写入日志
        writer.add_scalar("Test Loss", test_loss, epoch )
        writer.add_scalar("Accuracy", accuracy, epoch )
        # 刷新
        writer.flush()

        # 输出信息
        print("Test —— Average loss : {:.4f}, Accuracy : {:.3}\n".format(test_loss, accuracy))
        return test_loss, accuracy

# 定义函数，获取Tenerboard的writer
def tb_writer():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter('logdir/' + timestr)
    return writer

def main():
    # 3.定义超参数
    BATCH_SIZE = 1024 #每批处理数据的数量
    # DEVICE = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4.图片转换
    data_transforms = {
        "train":
            transforms.Compose([
                transforms.Resize(300),
                transforms.RandomResizedCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        "val":
            transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    }

    # 5.操作数据集
    # 5.1 数据集的路径
    data_path = "./chest_xray"
    # 5.2 加载数据集train 和 val
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                              data_transforms[x]) for x in["train", "val"]} #列表推导式
    # 5.3 为数据集创建一个迭代器，读取数据
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=BATCH_SIZE) for x in["train", "val"]}

    # 5.4 训练集和验证集大大小（图片的数量）
    data_size = {x: len(image_datasets[x]) for x in["train", "val"]} #列表推导式比for循环快得多，python列表里面做了优化

    # 5.5 获取标签的类别名称 NORMAL--正常，PNEUMONIA--感染
    target_names = image_datasets['train'].classes

    # 6 显示一个batch_size的图片
    # 6.1 读取若干张图片
    datas, targets = next(iter(dataloaders['train']))
    # 6.2 将若干张图片拼成一幅图像
    out = make_grid(datas, nrow=16, padding=10)
    # 6.3 显示图片
    image_show(out, title=[target_names[x] for x in targets])


if __name__ == '__main__':
    main()
