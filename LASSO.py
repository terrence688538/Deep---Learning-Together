import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

data = np.genfromtxt(r'D:\资源\数据科学\机器学习实战\回归\longley.csv', delimiter=",")
print(data)
x_data=data[1:,2:]
y_data=data[1:,1]
print(x_data)
print(y_data)

# 创建模型
model = linear_model.LassoCV()
model.fit(x_data, y_data)

# lasso系数
print(model.alpha_)
# 相关系数
print(model.coef_)
model.predict(x_data[-2,np.newaxis])

model.predict(x_data[-2,np.newaxis])