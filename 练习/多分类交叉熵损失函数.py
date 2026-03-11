import torch
from torch import nn

# 多分类交叉熵损失函数
def demo01():
    # 创建单个样本
    # 使用标签编码，1D
    y_true = torch.tensor([1,2,0],dtype=torch.int64)
    y_pred = torch.tensor([[0.1,2.0,1.0],[1.0,0.5,2.5],[2.0,0.8,0.3]],dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred,y_true)
    print(loss)
def demo2():
    # 创建多个样本
    # 使用one-hot编码，2D
    y_true =torch.tensor([[0,1,0],[0,0,1],[1,0,0]],dtype=torch.float32)
    # y_pred = torch.tensor([[1,20,3],[1,4,30],[25,3,1]],dtype=torch.float32)
    y_pred = torch.tensor([[0.1, 2.0, 1.0], [1.0, 0.5, 2.5], [2.0, 0.8, 0.3]], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred,y_true)
    print(loss)

if __name__ == '__main__':
    demo01()
    demo2()