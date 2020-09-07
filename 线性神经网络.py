import numpy as np
import matplotlib.pyplot as plt

#输入数据
X = np.array([[1,3,3],
              [1,4,3],
              [1,1,1],
              [1,0,2]])
#标签
Y = np.array([[1],
              [1],
              [-1],
              [-1]])

#权值初始化，3行1列，取值范围-1到1
W = (np.random.random([3,1])-0.5)*2

print(W)
#学习率设置
lr = 0.11
#神经网络输出
O = 0

def update():
    global X,Y,W,lr
    O = np.dot(X,W)
    W_C = lr*(X.T.dot(Y-O))/int(X.shape[0])
    W = W + W_C

for _ in range(100):
    update()#更新权值

    #正样本
    x1 = [3,4]
    y1 = [3,3]
    #负样本
    x2 = [1,0]
    y2 = [1,2]

    #计算分界线的斜率以及截距
    k = -W[1]/W[2]
    d = -W[0]/W[2]
    print('k=',k)
    print('d=',d)

    xdata = (0,5)

    plt.figure()
    plt.plot(xdata,xdata*k+d,'r')
    plt.scatter(x1,y1,c='b')
    plt.scatter(x2,y2,c='y')
    plt.show()

#解决异或的问题

import numpy as np
import matplotlib.pyplot as plt

#输入数据
X = np.array([[1,0,0,0,0,0],
              [1,0,1,0,0,1],
              [1,1,0,1,0,0],
              [1,1,1,1,1,1]])
#标签
Y = np.array([-1,1,1,-1])

#权值初始化，3行1列，取值范围-1到1
W = (np.random.random(6)-0.5)*2

print(W)
#学习率设置
lr = 0.11
#神经网络输出
O = 0
n = 0

def update():
    global X,Y,W,lr,n
    n +=1
    O = np.dot(X,W.T) # shape:(3,1)
    W_C = lr*((Y-O).dot(X))/int(X.shape[0])
    W = W + W_C

for i in range(1000):
    update()#更新权值
#正样本
x1 = [0,1]
y1 = [1,0]
#负样本
x2 = [0,1]
y2 = [0,1]

def calculate(x,root):
    a=W[5]
    b=W[2]+x*W[4]
    c=W[0]+x*W[1]+x*x*W[3]
    if root==1:
        return (-b+np.sqrt(b*b-4*a*c))/(2*a)
    if root==2:
        return (-b-np.sqrt(b * b-4 * a * c)) / (2 * a)


xdata = np.linspace(-1,2)
plt.figure()
plt.plot(xdata,calculate(xdata,1),'r')
plt.plot(xdata,calculate(xdata,2),'r')
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()

O=np.dot(X,W.T)
