# 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# ======================================1.准备数据==========================================
data = {
    'x':[1,3,2,1,3],
    'y':[14,24,18,17,27]
}
df = pd.DataFrame(data)
print(f'原始数据集：{df}')
print('-'*30)
# ======================================2.sklearn库模型训练==========================================
X = df['x'].values.reshape(-1,1)
y = df['y'].values
lr_model = LinearRegression()
lr_model.fit(X,y)
sklearn_slope = lr_model.coef_[0]
intercept_ = lr_model.intercept_
print(f'sklearn库模型训练的斜率：{sklearn_slope}')
print(f'sklearn库模型训练的截距：{intercept_}')
print(f'回归方程：y={sklearn_slope}x+{intercept_}')
print('-'*30)
# ======================================3.绘制可视化图像==========================================
#生成拟合直线x，y值（让直线更加的平滑，覆盖数据范围）
x_fit = np.linspace(df['x'].min()-0.5,df['x'].max()+0.5,100)
y_fit = sklearn_slope*x_fit+intercept_

#创建画布
plt.figure(figsize=(8,6))
plt.scatter(df['x'],df['y'],label='原始数据集')
plt.plot(x_fit,y_fit,label='sklearn线性回归模型')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('sklearn线性回归模型.png')