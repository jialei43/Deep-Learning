import torch
import torch.nn as nn

# 多分类交叉熵损失函数
def demo01():
    # 创建单个样本
    y_true = torch.tensor([1],dtype=torch.int64)
    y_pred = torch.tensor([[0.1,2.0,1.0]],dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred,y_true)
    print(loss)
def demo02():
    # 创建多个样本
    # 标签编码 1D
    # y_true = torch.tensor([1,2,0], dtype=torch.int64)
    # one-hot编码 2D
    y_true = torch.tensor([[0,1,0],[0,0,1],[1,0,0]],dtype=torch.float32)
    y_pred = torch.tensor([[0.1, 2.0, 1.0], [1.0, 0.5, 2.5], [2.0, 0.8, 0.3]], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred, y_true)
    print(loss)


# 二分类交叉熵损失函数
def demo03():
    # 创建多个样本
    # 二分类中y_true中 0代表负类  1代表正类  二分类中输出层只有一个神经元，所以输出层输出的概率为正类概率或负类概率
    y_true = torch.tensor([0,1,0,1],dtype=torch.float32)
    y_pred = torch.tensor([0.00001,0.9998,0.00005,0.95999],dtype=torch.float32)
    loss_fn = nn.BCELoss()
    loss = loss_fn(y_pred,y_true)
    print(loss)

# 回归任务损失函数
def demo04():

    y_true = torch.tensor([1,2,3,4],dtype=torch.float32)
    y_pred = torch.tensor([0.5,1.5,2.5,3.5],dtype=torch.float32)
    # MAE LOSS
    mae_loss = nn.L1Loss()
    loss = mae_loss(y_pred,y_true)
    print(loss)
    # MSE LOSS
    mse_loss = nn.MSELoss()
    loss = mse_loss(y_pred,y_true)
    print(loss)
    # SmoothL1Loss
    smooth_l1_loss = nn.SmoothL1Loss()
    loss = smooth_l1_loss(y_pred,y_true)
    print(loss)

if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
    demo04()