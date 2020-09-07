import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

data = np.genfromtxt(r'C:\Users\Mechrevo\Desktop\longley.csv', delimiter=",")
print(data)
x_data=data[1:,2:]
y_data=data[1:,1]
print(x_data)
print(y_data)

# 创建模型
model = linear_model.ElasticNetCV()
model.fit(x_data, y_data)

# 弹性网系数
print(model.alpha_)
# 相关系数
print(model.coef_)
model.predict(x_data[-2,np.newaxis])