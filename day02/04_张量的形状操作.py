"""
演示
    张量的形状操作
涉及到的API:
    reshape()       不改变张量内容/张量的连续性的前提下，改变张量的形状，不改变元素的顺序，从左到右从上到小的顺序
    unsqueeze()     在指定的轴/维度上增加一个维度（长度1），升维
    squeeze()       删除所有长度为1的维度，降维
    transpose()     交换维度，只能交换两个维度
    permute()       交换维度，可以交换多个维度
    view()          改变形状，要求是连续张量，张量中元素的顺序没有改变，
                    没有经过transpose或permute等操作
    contiguous()    将非连续张量转换为连续张量
    is_contiguous() 判断张量是否是连续张量

需要掌握：
    reshape()
    unsqueeze()
    permute()   交换多个维度/轴
    view()

"""

# 导包
import torch

# 1.定义函数，演示 张量的形状操作，改变形状
def demo01():
    # 0.设置随机种子
    torch.manual_seed(28)
    # 1.创建一个张量，(3,4)
    t1 = torch.randint(0,10,(3,4))
    print(f"t1:\n{t1}, shape: {t1.shape}")
    # 2.使用reshape()方法，改变张量的形状，(4,3)
    # t2 = t1.reshape(4,3)
    # t2 = t1.reshape(2,6)
    t2 = t1.reshape(-1,3)
    print(f"t2:\n{t2}, shape: {t2.shape}")
    # 3.使用unsqueeze()方法，升维，添加一个新的维度/轴
    t2 = t1.unsqueeze(dim=0).unsqueeze(0)
    print(f"t2:\n{t2}, shape: {t2.shape}")
    # 4.使用squeeze()方法，降维，删除所有长度为1的维度
    t3 = t2.squeeze()
    # t3 = t1.squeeze(dim=0)
    print(f"t3:\n{t3}, shape: {t3.shape}")

# 2.定义函数，演示 交换维度,view
def demo02():
    # 0.设置随机种子
    torch.manual_seed(28)
    # 1.创建一个张量，(3,4,5)
    t1 = torch.randint(0,10,(3,4,5))    # HWC
    print(f"t1:\n{t1}, shape: {t1.shape}")  # (3,4,5)
    # 2.使用permute()方法，交换维度 HWC -> CHW,改变了张量的连续性，也就是改变了张量的元素顺序
    t2 = t1.permute(2,0,1)  # (5,3,4)
    print(f"t2:\n{t2}, shape: {t2.shape}")
    # 2.1 使用is_contiguous()方法，判断张量是否是连续张量
    print(f"t2是否是连续张量：{t2.is_contiguous()}")

    # 3.使用transpose()方法，交换维度
    t3 = t1.transpose(2,0)  # (5,4,3)
    print(f"t3:\n{t3}, shape: {t3.shape}")
    # 4.使用view()方法，改变张量的形状，不改变张量的连续性/元素顺序，要求是连续张量
    # t4 = t1.view(5,3,4)
    # 4.1 使用contiguous()方法，将非连续张量转换为连续张量
    # t4 = t2.contiguous()
    # print(f"t4是否是连续张量：{t4.is_contiguous()}")
    t4 = t2.contiguous().view(3,4,5)
    # # t4 = t2.reshape(3,4,5)
    print(f"t4:\n{t4}, shape: {t4.shape}")

# 测试
if __name__ == '__main__':
    # demo01()
    demo02()