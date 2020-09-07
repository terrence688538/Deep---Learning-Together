import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_points=1000
vectors_set=[]
for i in range(num_points):
    x1=np.random.normal(0,0.55)
    y1=x1*0.1+0.3+np.random.normal(0,0.03)
    vectors_set.append([x1,y1])
x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]
plt.scatter(x_data,y_data,c='r')

#初始化参数 tf.variable
W=tf.Variable(tf.random_uniform([1],-1,1),name='W')         #生成1维的W矩阵，取值是[-1,1]
b=tf.Variable(tf.zeros([1]),name='b')
y=W*x_data+b

loss=tf.reduce_mean(tf.square(y-y_data),name='loss')              #reduce用于计算平均值
optimizer=tf.train.GradientDescentOptimizer(0.5)                  #梯度下降法步长这里是0.5
train=optimizer.minimize(loss,name='train')                       #最小化误差值
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

print('W=',sess.run(W),'b=',sess.run(b),'loss=',sess.run(loss))       #打印初始化的W,b,loss
for step in range(20):                                                 #执行20次训练
    sess.run(train)
    print('W=',sess.run(W),'b=',sess.run(b),'loss=',sess.run(loss))
plt.scatter(x_data,sess.run(W)*x_data+sess.run(b))