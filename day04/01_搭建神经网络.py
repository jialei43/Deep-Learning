"""
演示：
    搭建神经网络，这里是一个简单的全连接神经网络
全连接神经网络的组成：
    隐藏层1： nn.linear(3,3),权重初始化采用标准化的xavier初始化，激活函数使用sigmoid
    隐藏层2： nn.linear(3,2),权重初始化采用标准化的He初始化 激活函数采用的是relu
    out输出层： nn.linear(2,2),采用的是softmax做数据归一化，权重初始化采用标准化的xavier初始化

深度学习/神经网络训练的一般工作流
    1.准备数据集
    2.搭建神经网络
        继承nn.Module父类
        定义__init__方法，定义网络层
        定义forward方法，实现前向传播
    3.模式训练
    4.模型测试

"""
# 导包
import torch
from torch import nn
from torchsummary import summary # 计算机的模型参数，查看模型结构 conda install torchsummary


# 检查 MPS 是否可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


#1.定义模型类，搭建神经网络
class MyModel(nn.Module):
    """
    搭建神经网络
    继承nn.Module父类
    定义__init__方法，定义网络层，并进行参数初始化
    定义forward方法，实现前向传播
    """
    # 定义__init__方法，定义网络层,并进行参数初始化
    def __init__(self):
        # 1.初始化父类成员
        super(MyModel, self).__init__()
        # 2.定义网络层
        # 2.1 定义隐藏层1，输入维度3，输出维度3
        self.linear1 = nn.Linear(3, 3)
        # 2.2 定义隐藏层2，输入维度3，输出维度2
        self.linear2 = nn.Linear(3, 2)
        # 2.3 定义输出层，输入维度2，输出维度2
        self.linear3 = nn.Linear(2, 2)
        # 3.参数初始化
        # 3.1 隐藏层1的参数初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 3.2 隐藏层2的参数初始化
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        # 3.3 输出层的参数初始化
        nn.init.xavier_normal_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)


    # 2.定义forward方法，实现前向传播
    def forward(self,x):
        # 输入隐藏层1
        # x = self.linear1(x)
        # # 经过激活函数sigmoid
        # x = torch.sigmoid(x)
        # 合并起来写
        x = x.to(device)
        x = torch.sigmoid(self.linear1(x))
        # 输入隐藏层2
        # x = self.linear2(x)
        # # 经过激活函数reLu
        # x = torch.relu(x)
        # 合并起来写
        x = torch.relu(self.linear2(x))
        # 输入输出层
        x = torch.softmax(self.linear3(x),dim=-1)
        # 返回结果
        return x

# 模型训练
def train_model():
    # 1. 创建模型对象
    model = MyModel()
    print(model)
    model = model.to(device)
    # 2.查看模型的参数数量
    summary(model,(3,))


# 测试
if __name__ == '__main__':
    # model = MyModel()
    # print( model)
    train_model()
