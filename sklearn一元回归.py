# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:15:15 2019

@author: Mechrevo
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data=pd.read_csv(r'C:\Users\Mechrevo\Desktop\data.csv',header=None)           导入数据方法一
data = np.genfromtxt(r'C:\Users\Mechrevo\Desktop\data.csv', delimiter=",")
x_data=data[:,0]
y_data=data[:,1]
plt.scatter(x_data,y_data)
plt.show()
print(x_data.shape)

x_data = data[:,0, np.newaxis]
y_data = data[:,1, np.newaxis]
# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()

