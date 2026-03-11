import torch
from torch import nn

from torchsummary import summary


class MyModel(nn.Module):
    """
   搭建神经网络
   继承nn.Module父类
   定义__init__方法，定义网络层，并进行参数初始化
   定义forward方法，实现前向传播
   """

    def __init__(self):
        super(MyModel, self).__init__()
        # 1.定义网络层1
        self.linear1 = nn.Linear(3, 3)
        # 2.定义网络层2
        self.linear2 = nn.Linear(3, 2)
        # 3.定义输出层
        self.out= nn.Linear(2, 2)

        # 参数初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.ones_(self.linear2.bias)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))
        out = torch.softmax(self.out(x), dim=-1)
        return out

if __name__ == '__main__':
    model = MyModel()
    # 输入的张量是5个样本，每个样本有3个特征
    summary(model, input_size=(3,),batch_size=5)

    # 模拟数据
    x = torch.randn(5, 3)
    # model(x)等价与model.forward(x)
    out = model(x)
    print(out.shape)
    print(list(model.named_parameters()))
    print('-'*30)
    for name,parameter in model.named_parameters():
        print(name,parameter)


