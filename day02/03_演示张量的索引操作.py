"""
案例：
    演示 张量的索引操作
涉及到的API:
    简单行列索引:基于行列进行操作 t1[0,0] 第一个0代表行索引，第二个0代表列索引
    列表索引: t1[[0,1],[2,3]]
        1. 核心逻辑：点对点匹配
            当你使用两个列表（或 LongTensor）作为索引时，PyTorch 会将它们成对看作 (row, column) 坐标：
            第一个元素对：(0, 2) —— 第一行，第三列
            第二个元素对：(1, 3) —— 第二行，第四列
            结果： 它会返回一个包含这两个位置元素的一维张量，形状为 (2,)

    范围索引:基于行列进行匹配 t1[1::2, 0::2] 奇数行偶数列
    多维索引：基于维度/轴及向量的索引 t10[1, ::2, :3] 位置代表轴，数值代表索引 0轴1索引，1轴偶数索引，2轴的前三索引
    布尔索引
需要掌握的：
    简单行列索引 t1[0,0], t1[0]
    范围索引 t1[:2], t1[:2,2:4]
    多维索引 t1[0,0,0],t1[0,0,:2]
"""

import torch
# 1.定义函数，演示 张量的索引操作
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def demo01():
    torch.manual_seed(42)
    t1 = torch.randint(low=0,high=10,size=(3,4),device=torch.device('mps'))
    print(f't1:{t1},shape:{t1.shape},dtype:{t1.dtype},device:{t1.device},dim:{t1.ndim}')
    # 1.演示简单行列索引
    print(f't1[0,0]:{t1[0,0]},dim={t1[0,0].ndim},shape:{t1[0,0].shape},dtype:{t1[0,0].dtype},device:{t1[0,0].device}')
    print(f't1[0]:{t1[0]},dim={t1[0].ndim},shape:{t1[0].shape},dtype:{t1[0].dtype},device:{t1[0].device}')
    print(f't1[:,2]:{t1[:,2]},dim={t1[:,2].ndim},shape:{t1[:,2].shape},dtype:{t1[:,2].dtype},device:{t1[:,2].device}')
    # 2.演示列表索引
    #获取 （0，1）（1，2）位置的元素
    # print(f't1[0, [1,2]]:{t1[0, [1,2]]},dim={t1[0, [1,2]].ndim},shape:{t1[0, [1,2]].shape},dtype:{t1[0, [1,2]].dtype},device:{t1[0, [1,2]].device}')
    print(f't1[[0,1],[1,2]]:{t1[[0,1],[1,2]]},dim={t1[[0,1],[1,2]].ndim},shape:{t1[[0,1],[1,2]].shape},dtype:{t1[[0,1],[1,2]].dtype},device:{t1[[0,1],[1,2]].device}')
    # 获取(0,2),(1,3)位置的元素
    print(f't1[[0,1],[2,3]]:{t1[[0,1],[2,3]]},dim={t1[[0,1],[2,3]].ndim},shape:{t1[[0,1],[2,3]].shape},dtype:{t1[[0,1],[2,3]].dtype}')

    # 3.演示范围索引 切片
    # 获取前俩行数据
    print(f't1[:2]:{t1[:2]},ndim:{t1[:2].ndim},shape:{t1[:2].shape}')
    print('='*64)
    # 获取前俩列
    print(f't1[:, :2]:{t1[:, :2]},ndim:{t1[:, :2].ndim},shape:{t1[:, :2].shape}')
    print('='*64)
    # 获取偶数行
    print(f't1[1::2]:{t1[1::2]}')
    print('='*64)
    # 获取奇数行
    print(f't1[0::2]:{t1[0::2]}')
    print('='*64)
    # 获取偶数行的奇数列
    print(f'获取偶数行的奇数列 == t1[1::2, 0::2]:{t1[1::2, 0::2]}')
    print('='*64)

    # 4.演示多维索引
    t10 = torch.randint(low=0,high=10,size=(3,4,5),device=torch.device('mps'))
    print(f't10:\n {t10},shape:{t10.shape},dtype:{t10.dtype},device:{t10.device}')
    # 获取(0,0,0)位置的元素
    print(f't10[0,0,0]:{t10[0,0,0]},dim={t10[0,0,0].ndim},shape:{t10[0,0,0].shape}')
    print('='*64)
    # # 获取0轴索引1，1轴偶数索引，2轴前3个 的数据
    print(f't10[1, ::2, :3]:{t10[1, ::2, :3]}')
    print('='*64)
    # 6.演示 布尔索引
    # 获取第二行的大于5的所有元素
    t11 = t10[1, t10[1] > 5]
    print(f"t11: {t11}, ndim: {t11.ndim},shape: {t11.shape}")
    print('='*64)




if __name__ == '__main__':
    demo01()