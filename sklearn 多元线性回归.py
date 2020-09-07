import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=genfromtxt(r'D:\资源\数据科学\机器学习实战\回归\Delivery.csv', delimiter=",")
print(data)
x_data=data[:,0:2]
y_data=data[:,2]
print(x_data)
print(y_data)

model=linear_model.LinearRegression()
model.fit(x_data,y_data)
# 系数
print("coefficients:",model.coef_)
# 截距
print("intercept:",model.intercept_)
# 测试
x_test = [[102,4]]
predict = model.predict(x_test)
print("predict:",predict)

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)  # 点为红色三角形
x0 = x_data[:, 0]
x1 = x_data[:, 1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]
# 画3D图
ax.plot_surface(x0, x1, z)
# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')

# 显示图像
plt.show()

