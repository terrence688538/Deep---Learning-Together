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

alphas_to_test = np.linspace(0.001, 1)              #alphas_to_test = np.linspace(0.001, 1,100)           生成0.001-1的50个数（默认）
# 创建模型，保存误差值
model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)               #RidgeCVl岭回归加交叉验证法  alphas就是岭系数
model.fit(x_data, y_data)
# 岭系数
print(model.alpha_)
# loss值
print(model.cv_values_.shape)

# 画图
# 岭系数跟loss值的关系
plt.plot(alphas_to_test, model.cv_values_.mean(axis=0))
# 选取的岭系数值的位置
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)),'ro')
plt.show()
#预测
model.predict(x_data[2,np.newaxis])