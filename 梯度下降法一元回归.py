# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:32:37 2019

@author: Mechrevo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\Mechrevo\Desktop\data.csv',header=None)
x_data=data[0]
y_data=data[1]
plt.scatter(x_data,y_data)
plt.show()

b=0
k=0
lr=0.0001
epochs=50
#最小二乘法，计算损失函数
def compute_error(b,k,x_data,y_data):
    totalError=0
    for i in range(0,len(x_data)):
        totalError+=(y_data[i]-(b+k*x_data[i]))**2
    return totalError/float(len(x_data))/2
#计算梯度总和再求平均
def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    m=float(len(x_data))
    for i in range(epochs):
        b_grad=0
        k_grad=0
        for j in range(len(x_data)):
            b_grad+=(1/m)*(((k*x_data[j])+b)-y_data[j])
            k_grad+=(1/m)*x_data[j]*(((k*x_data[j])+b)-y_data[j])
        b=b-(lr*b_grad)
        k=k-(lr*k_grad)
        #每迭代5次画一次图
        if i % 5==0:
            print("epochs:",i)
            plt.plot(x_data, y_data, 'b.')
            plt.plot(x_data, k*x_data + b, 'r')
            plt.show()
    return b,k

print("Starting b = {0}, k = {1}, error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))  #format格式化输出
print("Running...")
b,k= gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k*x_data + b, 'r')
plt.show()
        
   
        
        