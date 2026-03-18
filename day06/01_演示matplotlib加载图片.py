"""
演示
    演示matplotlib的plt图像操作

图像分类：
    二值图像: 1通道, 0 或 1
    灰度图像: 1通道, 0 到 255
    索引图像: 1通道, 0 到 255, 像素点表示颜色表的索引
    RGB图像: 3通道，R红色、G绿色、B蓝色，象素点 0 到 255
    RGBA图像：4通道，R红色、G绿色、B蓝色、A透明度，0 到 255


涉及到API:
    plt.imshow() # 显示图片
    plt.imsave() # 保存图片
    plt.imread() # 读取图片
"""
import os

import numpy as np
from matplotlib import pyplot as plt


# 1.定义函数，显示 全黑 全白 图像
def demo01():
    # 创建全黑图片 numpy数组，实现全黑图片，HWC
    img1 = np.zeros((100, 100, 3))
    print(f'img1:{img1}, img1.shape:{img1.shape}')
    plt.imshow(img1)
    # 效果：
    # 不显示x轴 / y轴刻度
    # 不显示轴线
    # 图像只显示绘制内容（如曲线、图像本身），非常适合展示图片或可视化效果图
    # plt.axis('off')
    # plt.show()
    # 创建全白图片
    img2 = np.full((100, 100, 3),fill_value=255)
    print(f'img2:{img2}, img2.shape:{img2.shape}')
    plt.imshow(img2)
    # plt.axis('off')
    plt.show()

# 2.定义函数，保存图片
def demo02():
    img1 = np.zeros((100, 100, 3))
    print(f'img1:{img1}, img1.shape:{img1.shape}')
    plt.imshow(img1)
    os.makedirs('img', exist_ok=True)
    plt.imsave('img/img1.png',img1)


# 3. 定义函数，读取图片
def demo03():
    img = plt.imread('img/img.png')
    print(f'img:{img}, img.shape:{img.shape}')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # 保存图片
    os.makedirs('img', exist_ok=True)
    plt.imsave('img/img.jpg',img)

if __name__ == '__main__':
    # demo01()
    # demo02()
    demo03()