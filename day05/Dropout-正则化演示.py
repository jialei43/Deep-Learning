"""
正则化的方法：
    L1正则化：在损失函数中添加绝对值项|w|, 权重可以为0，相当于降维
    L2正则化：在损失函数中添加平方项w^2, 权重可以接近0，Adam/AdamW权重衰减，对离散值敏感

    Dropout随机失活：隐藏层的神经元以概率P输出置为0，未被失活的概率p会被放大
        1.训练时，model.train(),随机以概率p让一部分神经元失活，输出设置为0。没有失活的神经元的输出*1/(1-p)
        2.测试时，model.eval(),关闭dropout随机失活，让所有神经元一起参与

需要掌握：
    nn.Dropout(p=0.1)

"""
import torch.nn as nn
import torch

# 输入
x = torch.randn([1,4])

# 网络层
layer = nn.Linear(4,6)
# layer.eval() = 标记该层或模型为推理/测试模式 关闭 Dropout
# DropOut只在训练阶段生效，预测阶段不生效
# layer.eval()

# 输出
y = layer(x)
print(y)

#失活
dropout = nn.Dropout(0.5)
# dropout.eval() = 标记该层或模型为推理/测试模式 关闭 Dropout
# DropOut只在训练阶段生效，预测阶段不生效
dropout.eval()
for e in range(5):
    dropout_y = dropout(y)
    print(f'dropout_y:{dropout_y}')