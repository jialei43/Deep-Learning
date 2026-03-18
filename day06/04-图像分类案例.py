"""
演示
    CIFAR10图像分类案例，使用CNN网络来实现分类任务
神经网络训练的一般工作流/步骤：
    1.准备数据集
        CIFAR10数据集，需要torchvision自带的CIFAR10数据集，pip install torchvision
        包含50000张训练图片，10000张测试图片，10个类别，每个类别有6000张图片
        拆分训练集为训练集 和 验证集，从而得到训练集、验证集和测试集
    2.搭建神经网络
        使用CNN卷积神经网络
    3.模型训练
        创建数据加载器：张量 -> 数据集对象 -> 数据加载器
        设置损失函数和优化器
        开始训练,遍历轮数
            前向传播
            计算损失
            梯度清零
            反向传播
            更新参数
    4.模型测试
"""

#  导包
import os
import torch
import torch.nn as nn
from jieba import strdecode
from torchvision import transforms
from torchvision.datasets import CIFAR10    # CIFAR10数据集的API，用于获取或加载CIFAR10图片数据集
from torchvision.transforms import ToTensor, Compose  # 转换图片格式为pytorch的图片格式CHW
import torch.optim as optim # 优化器模块，提供各种优化器，比如SGD,Momentum,Adam,AdamW
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm   # 可视化训练过程
from yaml import compose

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题

# 0.全局配置 和 超参数
# 设备,优先级GPU > MPS > CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"当前设备：{DEVICE}")

# 1.准备数据集
def create_data():
    """
    创建数据集
    :return:
    """
    train_dataset = CIFAR10(root='data', train=True, transform=Compose([
    # transforms.Resize((128, 128)),     # 统一大小
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    # transforms.RandomResizedCrop(128, scale=(0.8, 1.0)), # 随机裁剪
    transforms.RandomVerticalFlip(), # 随机垂直翻转
    # transforms.RandomAffine(15), # 随机仿射
    transforms.RandomRotation(15),     # 随机旋转 ±15°
    transforms.ColorJitter(
        brightness=0.2,                # 亮度
        contrast=0.2,                  # 对比度
        saturation=0.2,                # 饱和度
        hue=0.1                        # 色相
    ),
    transforms.ToTensor(),             # 转为 tensor, 0~1
    # transforms.Normalize(               # 归一化
    #     mean=[0.485, 0.456, 0.406],    # ImageNet 均值
    #     std=[0.229, 0.224, 0.225]
    # )
]), download=False)
    test_dataset = CIFAR10(root='data', train=False, transform=Compose([ToTensor()]), download=False)
    # print(train_dataset.class_to_idx)
    # print(test_dataset.data.shape)
    # print(test_dataset.data[1])
    # print(test_dataset.targets[100])
    # plt.imshow(test_dataset.data[100])
    # plt.show()
    return train_dataset, test_dataset

# 2.搭建神经网络
class MyCNN(nn.Module):
    """
    搭建CNN网络
    """
    def __init__(self,num_classes=10):
        super(MyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 输出: 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 32x16x16

            nn.Conv2d(32, 64, 3, padding=1),  # 输出: 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 64x8x8

            nn.Conv2d(64, 128, 3, padding=1),  # 输出: 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 输出: 128x4x4
        )
        
        # 分类器：将特征图展平并通过全连接层映射到类别空间
        self.classifier = nn.Sequential(
            # 展平操作：将多维特征图转换为一维向量 (128*4*4=2048 维)
            nn.Flatten(),
            # 全连接层：2048 维输入映射到 256 维隐藏层
            nn.Linear(128 * 4 * 4, 256),
            # ReLU 激活函数：增加非线性
            nn.ReLU(),
            # Dropout 层：随机丢弃 40% 神经元，防止过拟合
            nn.Dropout(0.4),
            # 全连接层：256 维映射到 64 维
            nn.Linear(256, 64),
            # ReLU 激活函数：继续引入非线性
            nn.ReLU(),
            # Dropout 层：随机丢弃 20% 神经元，轻微正则化
            nn.Dropout(0.2),
            # 输出层：64 维映射到 num_classes 维（类别预测分数）
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.to(DEVICE)
        x = self.features(x)
        x = self.classifier(x)
        # 返回结果
        return x

def train(model, train_dataset, valid_dataset):
    """
    训练
    :param model:
    :param train_dataset:
    :return:
    """
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # 创建损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    EPOCHS = 200
    # 4.开始训练,遍历轮数
    # 初始化最优验证损失
    min_valid_loss = float("inf")
    # 保存最优损失损失，对应的模型参数
    best_model_state_dict = None
    # 最优指标
    best_acc = {}
    # ===== 用于绘图 =====
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(EPOCHS):
        # 0.设置为训练模式
        model.train()
        # 1.初始化 训练总损失，预测正确的样本数，训练总样本数
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # 2.遍历数据加载器，分批次训练
        for x, y in tqdm(train_loader):
            # x:(N,C,H,W)
            # y:(N,)
            # 0.迁移数据到设备
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # 1.前向传播
            y_pred = model(x)  # (N,10)
            # print(f"y_pred.shape: {y_pred.shape}")
            # 2.计算损失
            loss = loss_fn(y_pred, y)
            # 3.梯度清零
            optimizer.zero_grad()
            # 4.反向传播
            loss.backward()
            # 5.更新参数
            optimizer.step()
            # 6.计算训练损失和预测正确的样本数
            total_loss += loss.item() * x.size(0)
            total_correct += (y_pred.argmax(dim=-1) == y).sum().item()
            total_samples += x.size(0)
        # 3.计算当前轮次的平均训练损失和准确率
        avg_train_loss = total_loss / total_samples
        avg_train_acc = total_correct / total_samples
        # 4.在验证集上评估模型，计算当前轮次的模型的验证损失和验证准确率
        valid_loss, valid_acc = evaluate(valid_dataset, model)
        # 添加训练指标和验证指标到列表中
        train_losses.append(avg_train_loss)
        val_losses.append(valid_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(valid_acc)
        # 5.根据验证损失来保存最优模型
        if avg_train_loss < min_valid_loss:
            # 1.更新当前最优的验证损失
            min_valid_loss = avg_train_loss
            # 2.最优模型参数
            best_model_state_dict = model.state_dict()
            # 保存最优指标
            best_acc = {"avg_train_loss":avg_train_loss, "avg_train_acc":avg_train_acc, "valid_loss":valid_loss, "valid_acc":valid_acc}
            ...
        # 6.打印当前轮次的训练损失，训练准确率，验证损失，验证准确率
        print(f"{epoch + 1}/{EPOCHS} | "
              f"训练损失:{avg_train_loss:.4f} | "
              f"训练准确率：{avg_train_acc:.4f} | "
              f"验证损失:{valid_loss:.4f} | "
              f"验证准确率：{valid_acc:.4f}")

    os.makedirs("model", exist_ok=True)
    # 2.保存当前模型参数
    torch.save(best_model_state_dict, r"./model/mycnn.pth")
    print(f"保存最优模型，训练损失：{best_acc['avg_train_loss']:.4f} | 训练准确率：{best_acc['avg_train_acc']:.4f} | "
          f"验证损失:{best_acc['valid_loss']:.4f} | 验证准确率：{best_acc['valid_acc']:.4f}")
    # 5.返回结果
    return train_losses, train_accs, val_losses, val_accs


# 3.1 可视化训练过程
def plot_curves(train_losses, train_accs, val_losses, val_accs):
    # -----------------------------
    # 绘制训练/验证曲线
    # -----------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

@torch.no_grad()    # 关闭梯度计算
def evaluate(valid_dataset, model=None):
    # 1.创建数据加载器
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
    )
    # 2.检查是否有model，如果没有，则创建
    if model is None:
        model = MyCNN().to(DEVICE)
        # 加载模型参数文件
        model.load_state_dict(torch.load(r"./model/mycnn.pth"))
    # 3.设置为评估模式
    model.eval()
    # 4.设置损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 5.测试模型，遍历数据加载器
    # 1.初始化 总损失，预测正确的样本数，总样本数
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    # 2.遍历数据加载器，分批次训练
    for x, y in tqdm(valid_dataloader):
        # x:(N,C,H,W)
        # y:(N,)
        # 0.迁移数据到设备
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # 1.前向传播
        y_pred = model(x)  # (N,10)
        # print(f"y_pred.shape: {y_pred.shape}")
        # 2.计算损失
        loss = loss_fn(y_pred, y)
        # # 3.梯度清零
        # optimizer.zero_grad()
        # # 4.反向传播
        # loss.backward()
        # # 5.更新参数
        # optimizer.step()
        # 6.计算损失和预测正确的样本数
        total_loss += loss.item() * x.size(0)
        total_correct += (y_pred.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)
    # 3.计算验证的平均损失和准确率
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    # 4.返回结果
    return avg_loss, avg_acc


if __name__ == '__main__':
    train_dataset,test_dataset = create_data()
    print(f'len(train_dataset): {len(train_dataset)}')
    print(f'len(test_dataset): {len(test_dataset)}')
    # 2.创建神经网络对象
    model = MyCNN()
    model = model.to(DEVICE)
    print(model)
    summary(model, (3, 32, 32))
    # 3.模型训练
    train_losses, train_accs, val_losses, val_accs = train(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=test_dataset,
    )
    # 可视化训练过程
    # plot_curves(train_losses, train_accs, val_losses, val_accs)
    # 4.模型测试
    test_loss, test_acc = evaluate(test_dataset)
    print(f"测试损失:{test_loss:.4f}，测试准确率：{test_acc:.4f}")
