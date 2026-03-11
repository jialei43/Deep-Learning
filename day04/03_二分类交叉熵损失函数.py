"""
演示
    二分类交叉熵损失函数
    多分类交叉熵损失函数

介绍
    损失函数 也叫 成本函数，目标函数，代价函数，误差函数，用来 衡量模型好坏，也就是模型预测表现
    分类问题
        多分类选择多分类交叉熵损失函数，nn.CrossEntropLoss
        二分类任务可以使用二分类交叉熵，nn.BCELoss
多分类交叉熵损失
    公式：CrossEntropLoss = -Σylog(softmax(y_logits))
    其中：y_logits是模型的原始输出/预测分数
        softmax(y_logits)是模型输出的概率分布，也就是每个类别的预测概率，最大概率对应预测类别
    注意：
        多分类交叉熵CrossEntropLoss里面有softmax运算，所以模型的输出层不需要softmax。
        但是如果要获取预测概率，则需要在模型输出的原始预测分数后面经过softmax

二分类交叉熵损失
    公式：BCELoss = -ylog(y_hat)-(1-y)log(1-y_hat)
    其中：y_hat是模型的预测值，也就是经过sigmoid之后的预测概率，对应正类
        y 是真实值，0或1，
        负类：标签编码0，one-hot编码[1,0],正类：标签编码1，one-hot[0,1]
    注意：
        二分类交叉熵损失BCELoss里面没有sigmoid，所以模型输出层最后需要添加sigmoid

需要掌握：
    loss_fn = nn.CrossEntropLoss
    loss = loss_fn(y_logits, y_true)
    二分类任务建议用多分类交叉熵损失函数
    可以使用二分类交叉熵损失BCELoss，也可以使用多分类交叉熵损失CrossEntropLoss

"""
# 导包
import torch
from torch import nn

# 1.定义函数，演示二分类交叉熵损失函数
def demo01():
    # 1.创建 样本的真实值，假设是一个二分类任务
    # 1.1使用 one-hot编码,2D
    # y_true = torch.tensor([[0,1]],dtype=torch.float32) # 代表类别为1，形状(batch_size,*)
    # 1.2使用标签编码，1D
    y_true = torch.tensor([0,1,0,1],dtype=torch.float32)
    # 2.创建 模型的预测分数y_logits
    y_logits = torch.tensor([10,-10,-100,2.5],dtype=torch.float32)
    #经过sigmoid，转换为概率，这里才是模型输出的预测值
    y_pred_prob = torch.sigmoid(y_logits)
    print(f'概率分布：{y_pred_prob}')
    # 3.创建损失函数
    loss_fn = nn.BCELoss()
    # 4.计算损失
    loss = loss_fn(y_pred_prob,y_true)
    print(f'二分类交叉熵损失函数计算结果为：{loss}')


# 2.定义函数，演示 多分类交叉熵损失函数 实现 二分类任务的计算损失
def demo02():
    # 1.创建 样本的真实值，假设是一个三分类任务
    # 1.1 使用one-hot编码，2D
    # y_true = torch.tensor([[0,1,0]], dtype=torch.float32)    # 代表类别为1，形状(batch_size,*)
    # 1.2 使用标签编码,1D
    y_true = torch.tensor([0, 1, 0, 1], dtype=torch.long)    # 形状(batch_size,)
    # 2.创建 模型的预测分数y_logits，[10, -10, -100, 2.5]
    y_logits = torch.tensor([[0,10],[0,-10],[0,-100],[1,3.5]])    # (batch_size, 2)
    # 3.创建 多分类交叉熵损失函数对象
    loss_fn = nn.CrossEntropyLoss()
    # 4.计算损失
    loss = loss_fn(y_logits,y_true)
    print(f"多分类交叉熵损失函数：{loss}")

if __name__ == '__main__':
    demo01()
    demo02()