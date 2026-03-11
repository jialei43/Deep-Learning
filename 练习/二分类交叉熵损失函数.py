import torch
from torch import nn


def demo01():
    # 创建多个样本
    # 使用标签编码，1D
    y_true = torch.tensor([1,0,1,0],dtype=torch.float32)
    logits = torch.tensor([10,1,10,1],dtype=torch.float32)
    y_pred = torch.sigmoid(logits)
    loss_fn=nn.BCELoss()
    loss=loss_fn(y_pred,y_true)
    print(loss)



if __name__ == '__main__':
    demo01()