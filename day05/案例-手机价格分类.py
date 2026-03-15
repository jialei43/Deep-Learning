import os
import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchsummary import summary

# 检查 MPS 是否可用
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"当前设备：{device}")


# 构建数据集
def create_dataset():
    # 使用pandas读取数据
    data = pd.read_csv('../data/手机价格预测.csv')
    print(data.info())
    print(data.head(5))
    # 特征值和目标值
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype('float32')
    y = y.astype('float32')

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)

    # ndarray 转 tensor   也可以使用torch.from_numpy() 内存共享 torch.tensor() 不共享 === torch.from_numpy().copy
    x_train = torch.tensor(x_train.values).to(dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train.values).to(dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test.values).to(dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test.values).to(dtype=torch.float32, device=device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # print(list(train_dataset))
    return train_dataset, test_dataset


# 创建神经网络
class LinearRegression(torch.nn.Module):
    def __init__(self, input_features=20, output_features=4):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_features, 256)
        self.dp1 = torch.nn.Dropout(p=0.3)
        self.linear2 = torch.nn.Linear(256, 128)
        self.dp2 = torch.nn.Dropout(p=0.2)
        self.linear3 = torch.nn.Linear(128, 64)
        self.dp3 = torch.nn.Dropout(p=0.1)
        self.outer = torch.nn.Linear(64, output_features)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        self.dp1(x)
        x = torch.relu(self.linear2(x))
        self.dp2(x)
        x = torch.relu(self.linear3(x))
        self.dp3(x)
        x = self.outer(x)
        return x


def train_model(train_dataset, input_dim, out_dim, epochs):
    # 加载数据集
    # 参数1：数据集对象；参数2：批次大小；参数3：是否打乱数据，训练时打乱，测试时不打乱；参数4: 是否删除最后一个不完整的批次数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"训练集数量：{len(train_loader)}")
    # 2.创建模型对象
    model = LinearRegression(input_dim, out_dim)
    # 2.模型使用GPU
    model = model.to(device)
    # 3.定义损失函数和优化器
    loss_fn = torch.nn.CrossEntropyLoss()
    # 4.创建优化器对象
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # 学习率衰减策略
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # 定义变量，记录 每个轮次的 训练损失 和 准确率
    train_losses = []
    train_accs = []
    # 存储准确率和模型对应的参数，方便后面保存损失函数最小的对应的模型
    per_round_model_state_dict=[]
    # 5.模型训练
    for epoch in range(epochs):
        # 1.设置训练模式model.train()
        model.train()
        # 2.初始化 训练的总损失，总样本数，预测正确的样本数
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        # 获取当前时间
        start_time = time.time()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # 1.前向传播
            y_pred = model(x_batch)
            # 2.计算损失
            loss = loss_fn(y_pred, y_batch)
            # 3.反向传播
            optimizer.zero_grad()
            loss.backward()
            # 4.更新参数
            optimizer.step()
            # 5.更新训练指标，训练损失，训练样本数，预测正确的样本数
            # 训练损失
            total_loss += loss.item() * x_batch.shape[0]
            # 训练样本数
            total_samples += x_batch.shape[0]
            # 预测正确的样本数  沿着维度 1（列方向）找到最大值的索引位置
            total_correct += (y_pred.argmax(1) == y_batch).sum().item()
        #  6.学习率衰减 (momentus动量法)
        # lr_scheduler.step()
        # 4.计算当前轮次的训练损失和训练准确率
        avg_train_loss = total_loss / total_samples
        avg_train_acc = total_correct / total_samples
        # 获取训练损失和准确率
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        per_round_model_state_dict.append({"epoch": epoch,"avg_train_loss": avg_train_loss, "model_state_dict": model.state_dict()})
        end_time = time.time()
        epoch_time = end_time - start_time
        # 5.打印训练损失和训练准确率
        print(f"{epoch + 1}/{epochs} | "
              f"训练损失：{avg_train_loss:.4f} | "
              f"训练准确率：{avg_train_acc:.4f} | "
              f"训练时间：{epoch_time:.2f}s")
    # 5.保存模型文件
    # 选择训练过程中最优的模型 测试中发现训练过程中准确率最高的模型，在测试中表现并不是最好的，反而是最后一次模型表现最好
    # 选择训练过程中损失函数值最小的模型
    max_item = max(per_round_model_state_dict, key=lambda x: x["avg_train_loss"])
    print(f'最优模型：{max_item}')
    # 新建文件夹model
    os.makedirs("model", exist_ok=True)
    # 保存模型参数文件，因为在不同设备上迁移更加方便，如果把模型结构也保存了，那么模型在不同设备上迁移困难
    # print(f"模型参数：{model.state_dict()}")
    torch.save(max_item["model_state_dict"], "model/phone_price_classifier.pth")
    # torch.save(model.state_dict(), "model/phone_price_classifier.pth")
    # 返回训练损失和训练准确率  为可视化做数据准备
    return train_losses, train_accs

# 3.1 可视化训练过程
def plot_train_process(train_losses, train_accs):
    plt.figure(figsize=(12, 6))
    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    # plt.show()
    # 绘制训练准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="train_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 4.模型测试-在测试集上测试
# 参数1：测试集
@torch.no_grad()   # 关闭梯度计算
def evaluate(test_dataset, input_dim=20, output_dim=4):
    # 1.创建数据加载器对象
    # 参数1：数据集对象；参数2：批次大小；参数3：是否打乱数据，训练时打乱，测试时不打乱；参数4: 是否删除最后一个不完整的批次数据
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    # 2.创建神经网络模型对象
    model = LinearRegression(input_dim, output_dim)
    # 迁移模型到设备
    model = model.to(device)
    # 3.加载模型参数，将模型参数文件的模型参数加载到示例化的模型对象中
    model.load_state_dict(torch.load("model/phone_price_classifier.pth"))
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 4.模型测试，设置为评估模式model.eval()，告诉模型，现在是训练/评估，不要开启dropout等特殊网络层
    model.eval()
    # 定义变量，记录测试 的总损失，总样本数，预测正确的样本数
    test_loss = 0.0
    total_samples = 0
    total_correct = 0
    # 遍历批次，也就是遍历数据加载器
    for x, y in test_dataloader:
        # x:(batch_size,input_dim)
        # y:(batch_size,)
        # 0.迁移数据到device
        x = x.to(device)
        y = y.to(device)
        # 1.前向传播
        y_pred = model(x)  # (batch_size,4)
        # 2.计算损失
        loss = loss_fn(y_pred, y)
        # 梯度清零
        # 反向传播
        # 更新参数
        # 3.更新指标，损失，样本数，预测正确的样本数
        batch_size = x.shape[0]
        # 更新总损失
        test_loss += loss.item() * batch_size  # 当前批次的总损失=平均损失*当前批次的批次大小
        # 更新样本数
        total_samples += batch_size
        # 更新预测正确的样本数
        total_correct += (y_pred.argmax(dim=-1) == y).sum().item()
        ...
    # 计算测试损失和测试准确率
    avg_test_loss = test_loss / total_samples
    avg_test_acc = total_correct / total_samples
    # 返回结果
    return avg_test_loss, avg_test_acc


if __name__ == '__main__':
    # 1.准备数据集
    train_dataset, test_dataset = create_dataset()
    print(f"训练集数量：{len(train_dataset)}")
    print(f"测试集数量：{len(test_dataset)}")
    print(f"train_dataset:{train_dataset}, test_dataset:{test_dataset}")
    # 2.创建模型对象
    # model = LinearRegression()    # 默认模型对象在CPU上
    # model = model.to(device)    # 将模型对象移动到device上，比如GPU
    # print(model,)
    # summary 不支持GPU，只能使用CPU
    # summary(model, input_size=(20,),device= "cpu")
    # train_losses, train_accs = train_model(train_dataset, 20, 4, 500)
    # 可视化训练过程
    # plot_train_process(train_losses, train_accs)
    # 4.模型测试
    avg_test_loss, avg_test_acc = evaluate(test_dataset, input_dim=20, output_dim=4)
    print(f"测试损失：{avg_test_loss:.4f} | 测试准确率：{avg_test_acc:.4f}")
