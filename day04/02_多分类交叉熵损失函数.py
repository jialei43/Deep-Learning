"""
演示
    多分类交叉熵损失函数

介绍
    损失函数 也叫 成本函数，目标函数，代价函数，误差函数，用来 衡量模型好坏，也就是模型预测表现
    分类问题
        多分类选择多分类交叉熵损失函数，nn.CrossEntropLoss
多分类交叉熵损失
    公式：CrossEntropLoss = -Σylog(softmax(y_logits))
    其中：y_logits是模型的原始输出/预测分数
        softmax(y_logits)是模型输出的概率分布，也就是每个类别的预测概率，最大概率对应预测类别
    注意：
        多分类交叉熵CrossEntropLoss里面有softmax运算，所以模型的输出层不需要softmax。
        但是如果要获取预测概率，则需要在模型输出的原始预测分数后面经过softmax

需要掌握：
    loss_fn = nn.CrossEntropLoss()
    loss = loss_fn(y_logits, y_true)

"""
# 导包
import torch
from torch import nn

# 1.定义函数，演示 多分类交叉熵损失函数
def demo01():
    # 1.创建 样本的真实值，假设是一个三分类任务
    # 1.1使用 one-hot编码,2D
    # y_true = torch.tensor([[0,1,0]],dtype=torch.float32) # 代表类别为1，形状(batch_size,*)
    # 1.2使用标签编码，1D
    y_true = torch.tensor([1],dtype=torch.int64)
    # 2.创建 模型的预测分数y_logits
    y_logits = torch.tensor([[0.1,2.0,1.0]],dtype=torch.float32)
    # 3.创建损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 4.计算损失
    loss = loss_fn(y_logits,y_true)
    print(f'多分类交叉熵损失函数计算结果为：{loss}')
    # 5.将y_logits转换为概率分布，经过softmax
    y_pred_prob = torch.softmax(y_logits,dim=-1)
    print(f'概率分布：{y_pred_prob}')
    # 获取最大的概率值,手动计算交叉熵损失函数结果
    # 这里不应该用max()来进行损失函数结果计算，应该使用索引进行计算
    # my_loss = -torch.log(y_pred_prob.max())
    my_loss = -torch.log(y_pred_prob[0][y_true])
    print(f'手动计算交叉熵损失函数结果：{my_loss}')

# 2.定义函数，演示 多分类交叉熵损失函数（多样本数）
def demo02():
    # 1. 创建样本的真实值，假设是一个三分类任务，共3个样本
    # 1.1 使用标签编码，1D，形状(batch_size,)
    y_true = torch.tensor([1, 2, 0], dtype=torch.int64)  # 3个样本的真实类别：第1个样本是类别1，第2个样本是类别2，第3个样本是类别0

    # 2. 创建模型的预测分数y_logits，形状(batch_size, num_classes)
    y_logits = torch.tensor([
        [0.1, 2.0, 1.0],  # 样本1的预测分数
        [1.0, 0.5, 2.5],  # 样本2的预测分数
        [2.0, 0.8, 0.3]  # 样本3的预测分数
    ], dtype=torch.float32)

    # 3. 创建损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 4. 计算损失（默认会返回batch的平均损失）
    loss = loss_fn(y_logits, y_true)
    print(f'多分类交叉熵损失函数计算结果（平均损失）：{loss:.4f}')

    # 5. 计算每个样本的softmax概率
    y_pred_prob = torch.softmax(y_logits, dim=-1)
    print(f'\n概率分布：')
    print(y_pred_prob)

    # 6. 计算每个样本的交叉熵损失
    sample_losses = []
    # for i in range(len(y_true)):
    #     sample_loss = -torch.log(y_pred_prob[i][y_true[i]])
    #     sample_losses.append(sample_loss)
    #     print(f'样本{i + 1}的损失：{sample_loss:.4f}')

    # for循环可以被简化代替
    # 5.2 使用gather一次性提取每个样本真实类别对应的概率
    # y_true.unsqueeze(1) 将形状从(3,)变为(3,1)，用于gather索引
    sample_losses = y_pred_prob.gather(1, y_true.unsqueeze(1)).squeeze()

    # 7. 计算平均损失
    # my_loss_avg = torch.tensor(sample_losses).mean()
    my_loss_avg = -torch.log(sample_losses).mean()
    print(f'\n手动计算的平均交叉熵损失：{my_loss_avg:.4f}')

    # 8. 验证两种计算方式是否一致
    print(f'\n两种计算结果是否一致：{torch.allclose(loss, my_loss_avg)}')

    # 9. 详细解释
    print(f'\n{"=" * 50}')
    print("结果解释：")
    print("=" * 50)
    print(f"真实标签：{y_true.numpy()}")
    print(f"预测分数：\n{y_logits.numpy()}")

    # 找出每个样本的最大概率类别
    pred_classes = torch.argmax(y_pred_prob, dim=-1)
    print(f"\n预测类别：{pred_classes.numpy()}")

    print(f"\n详细计算过程：")
    for i in range(len(y_true)):
        prob_for_true = y_pred_prob[i][y_true[i]].item()
        print(f"\n样本{i + 1}:")
        print(f"  真实类别: {y_true[i]}")
        print(f"  真实类别对应的概率: {prob_for_true:.4f}")
        print(f"  损失计算: -log({prob_for_true:.4f}) = {-torch.log(torch.tensor(prob_for_true)):.4f}")
        print(f"  预测概率分布: {y_pred_prob[i].numpy()}")

    print(f"\n最终结果：")
    print(f"  PyTorch计算的交叉熵损失: {loss:.4f}")
    print(f"  手动计算的平均交叉熵损失: {my_loss_avg:.4f}")
    print(f"  注：CrossEntropyLoss默认计算的是所有样本的平均损失")

if __name__ == '__main__':
    # demo01()
    demo02()