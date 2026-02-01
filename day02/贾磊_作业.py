import torch
# - 1.使用torch.tensor()创建两个形状为(3,4)的张量t1和t2，计算(t1+t2-2)*2, 并打印。
t1 = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
t2 = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print((t1+t2-2)*2)
print('='*64)
# - 2.使用torch.arange创建一个形状为(3,4)的张量t1,创建一个形状为(3,4)的随机张量t2，计算t1*t2,然后将t2转换形状为(4,3)，注意不要改变t2中元素顺序，计算t1@t2,并打印。
t1 = torch.arange(1,13).reshape(3,4)
t2 = torch.arange(2,14).reshape(3,4)
print(t1*t2)
t2 = t2.reshape(4,3)
print(t1@t2)
print('='*64)
# - 3.使用torch.randint()创建一个形状为(3,4)的随机整数张量,求所有元素的均值 和 第1个轴的均值，并打印。(注意元素类型转换)
# 设置随机种子
torch.manual_seed(22)
t1 = torch.randint(low=0,high=10,size=(3,4),device=torch.device('mps'))
print(t1.float().mean())
# 获取第一个轴的均值
print(t1.float().mean(dim=0))
print('='*64)
# - 4.使用torch.randint()创建一个形状为(3,4)的张量，获取第2个轴的 索引为1的所有数据，然后获取第二个轴的 索引为1的 大于5的元素 对应的 第一个轴的数据，并打印。
t1 = torch.randint(low=0,high=10,size=(3,4),device=torch.device('mps'))
print(t1)
print(t1[:,1])
# print(t1[:,1][t1[:,1]>5])
print(t1[t1[:,1]>5])
print('='*64)
# - 5.创建两个形状为(3,4)的随机张量t1和t2，都转换形状为(4,3), 然后按0轴进行拼接，得到形状为(8,3)的张量，并打印。
t1 = torch.randint(low=0,high=10,size=(3,4),device=torch.device('mps'))
t2 = torch.randint(low=0,high=10,size=(3,4),device=torch.device('mps'))
t1 = t1.reshape(4,3)
t2 = t2.reshape(4,3)
print(torch.cat([t1,t2],dim=0))
print('='*64)

# - 6.创建一个形状为(3)的随机张量w，开启requires_grad, 计算loss = w**2 + 20, 使用loss.mean().backward()获取梯度，并打印梯度w.grad。（尝试手动计算梯度）
w = torch.randint(low=0,high=10,size=(3,),device=torch.device('mps'),requires_grad=True,dtype=torch.float32)
print(w)
loss = w**2 + 20
loss.mean().backward()
print(w.grad)