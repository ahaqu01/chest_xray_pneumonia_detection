# 1.导入必要的库
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms,models, utils
from torchsummary import summary #可视化训练
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from PIL import Image

# 2.分为train,val,test定义transform
image_transform = {
    'train':transforms.Compose([
        transforms.RandomSizedCrop(size=300, scale=(0.8, 1.1)), #功能：随机长宽比裁剪原始图片，表示随机crop出来的图片会在0.8-1.1倍之间
        transforms.RandomRotation(degrees=10), #功能：根据degreen随机旋转一定角度，则表示（-10,+10）度之间随机旋转
        transforms.ColorJitter(0.4, 0.4, 0.4), #功能：修改亮度、对比度、饱和度
        transforms.RandomHorizontalFlip(), #功能：水平翻转
        transforms.CenterCrop(size=256), #功能：根据给定的size从中心裁剪，size--若为squence,则为(w,h),本次w=256,h=256
        transforms.ToTensor(), # numpy-->tensor
        # 功能：对数据按通道进行标准化（RGB）,即先减均值，再除以标准差
        transforms.Normalize([0.485, 0.456, 0.406], # mean
                             [0.229, 0.224, 0.225]) # std
    ]),
    'val':transforms.Compose([
        transforms.Resize(size=300),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.226])
    ]),
    'test':transforms.Compose([
        transforms.Resize(size=300),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.226])
    ])
}

# 加载数据集

# 数据集目录所在目录路径
data_dir = './chest_xray/'
# train路径
train_dir = data_dir + 'train/'
# val路径
val_dir = data_dir + 'val/'
# test路径
test_dir = data_dir + 'test/'

# 从文件中读取数据
datasets = {
    'train':datasets.ImageFolder(train_dir, transform=image_transform['train']), # 读取train中的数据集，并transform
    'val':datasets.ImageFolder(val_dir, transform=image_transform["val"]), # 读取val中的数据集，并transform
    'test':datasets.ImageFolder(test_dir, transform=image_transform["test"]) #读取test中的数据集，并transform
}

# 定义BATCH_SIZE
BATCH_SIZE = 128

# DataLoader: 创建iterator, 按批读取数据
dataloaders = {
    'train':DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True), #训练集
    'val':DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=True), #验证集
    'test':DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=True) #测试集
}

# 创建label的键值树
LABEL = dict((v,k)  for k, v in datasets['train'].class_to_idx.items())
print(LABEL)

# train 简介
print(dataloaders['train'].dataset)
print(dataloaders['train'].dataset.classes) # train下的类别
print(dataloaders['train'].dataset.root) # train的路径

# 肺部正常的图片
files_normal = os.listdir(os.path.join(str(dataloaders['train'].dataset.root), 'NORMAL'))
#print(files_normal)
# 肺部感染的图片
files_pneumonia = os.listdir(os.path.join(str(dataloaders['train'].dataset.root), 'PNEUMONIA'))

# val 简介
print(dataloaders['val'].dataset)
print(dataloaders['val'].dataset.classes)
print(dataloaders["val"].dataset.root)

# test 简介
print(dataloaders['test'].dataset)

# 提示需要安装库： tb-nightly
# 导入SummaryWriter
from torch.utils.tensorboard import SummaryWriter
#  SummaryWriter()向事件文件写入事件和概要

# 定义日志路径
log_path = 'logdir/'

# 定义函数：获取tensorboard writer
def tb_writer():
    timestr = time.strftime("%Y%m%d_%H%M%S" ) #事件格式
    writer = SummaryWriter(log_path+timestr) #写入日志
    return writer

writer = tb_writer()

# 第一种方法：显示部分图片集
images, labels = next(iter(dataloaders["train"])) #获取到一批数据

# 定义图片显示方法
def imshow(img):
    img = img / 2 + 0.5 # 逆正则化
    np_img = img.numpy() # tensor-->numpy
    plt.imshow(np.transpose(np_img, (1, 2, 0))) #改变通道顺序
    plt.show()

grid = utils.make_grid(images) # make_grid 的作用是将若干幅图像拼成一幅图
imshow(grid)

# 在summary中添加图片数据
writer.add_image("X-Ray grid", grid, 0 ) # add_image详解见函数

writer.flush() # 把事件文件写入到磁盘中

# 获取一张图片tensor
print(dataloaders['train'].dataset[4]) # 第5张图片，返回：tensor, label

# 第2种方法，显示一张图片
def show_sample(img, label):
    print("Label :", dataloaders['train'].dataset.classes[label]) #输出标签
    img = img.numpy().transpose(1, 2, 0) #改变shapeshunx
    mean = np.array([0.485, 0.456, 0.406]) # 均值
    std = np.array([0.229, 0.224, 0.225])  # 标准差
    img = img * std + mean # 逆向复原
    img = np.clip(img, 0, 1) # np.clip() 将inp中的元素值限制在(0,1)之间，最小值为0，最大值为1。小于min的等于min，大于max等于max
    plt.imshow(img)
    plt.axis("off") # 关闭坐标轴

show_sample(*dataloaders['train'].dataset[4]) # 显示第5张图片


# 第3种方法：显示一张图片
def show_image(img):
    plt.figure(figsize=(8, 8))  # 显示大小
    plt.imshow(img)  # 显示图片
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 读取图片
one_img = Image.open(dataloaders['train'].dataset.root + 'NORMAL/IM-0239-0001.jpeg')

# 调用函数
show_image(one_img)

# 记录错误分类的图片
def misclassified_image(pred, writer, target, images, output, epoch, count=10):
    misclassified = (pred != target.data) # 判断是否一致
    for index, images_tensor in enumerate(images[misclassified][:count]):
        img_name = 'Epoch:{}-->Predict:{}-->Actual:{}'.format(epoch, LABEL[pred[misclassified].tolist()[index]],
                                                              LABEL[target.data[misclassified].tolist()[index]])
        writer.add_image(img_name, images_tensor, epoch)

# 自定义池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        size = size or (1, 1)
        #自适应算法能够自动帮助我们计算核的大小和每次移动的步长
        self.avgPooling = nn.AdaptiveAvgPool2d(size) # 自适应平均池化
        self.maxPooling = nn.AdaptiveMaxPool2d(size) # 最大池化

    def forward(self, x):
        # 拼接avg和max
        return torch.cat(self.maxPooling(x), self.avgPooling(x), dim=1)

# 迁移学习：获取预训练模型，并替换池化层和全连接层
def get_model():
    # 获取预训练模型 resNet50
    model = models.resnet50(pretrained=True)
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
    # 替换最后2层：池化层和全连接层。微调模型：替换ResNet最后的两层网络，返回一个新的模型
    # 池化层
    model.avgpool = nn.AdaptiveConcatPool2d()
    # 全连接层
    model.fc = nn.Sequential(
        nn.Flatten(), # 拉平
        nn.BatchNorm1d(4096), # 加速神经网络的收敛过程，提高训练过程中的稳定性
        nn.Dropout(0.5), # 丢掉部分神经元，防止过拟合
        nn.Linear(4096, 512), # 全连接层
        nn.ReLU(), # 激活函数
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 2), # 两个输出，正常、感染
        nn.LogSoftmax(dim=1) # 损失函数：将input转换成概率分布的形式，输出2个概率
    )

    return model

# 定义训练函数
def train_val(model, device, train_loader, val_loader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss = 0.0
    val_loss = 0.0
    val_acc = 0
    for batch_id, (images, labels) in enumerate(train_loader):
        # 部署到device上
        images, labels = images.to(device), labels.to(device)
        # 梯度置0
        optimizer.zero_grad()
        # 模型输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        total_loss += loss.item() * images.size(0)
    # 平均训练损失
    train_loss = total_loss / len(train_loader.dataset)
    # 写入到writer中
    writer.add_scalar("Train Loss", train_loss, epoch )
    # 写入到磁盘
    writer.flush()




