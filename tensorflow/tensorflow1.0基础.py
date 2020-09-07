import tensorflow as tf
import numpy as np
#tensorflow中创建变量要用Variable的形式
w=tf.Variable([[0.5,1]])
x=tf.Variable([[2,0],[1,0]])
y=tf.matmul(w,tf.cast(x,tf.float32))#这是由于两个相乘矩阵类型不匹配，调试一下发现x矩阵为tf.float64,W矩阵为tf.float32，改类型用tf.cast()函数

init_op=tf.global_variables_initializer()  #对上面写的东西做全局初始化

with tf.Session() as sess:                   # tensorflow是在session中进行存储然后调出进行运算
    sess.run(init_op)
    print(y.eval())
    print(x.eval())

a=tf.zeros([3,4],dtype=tf.int32)
b=tf.ones([2,3],dtype=tf.float32)
c=tf.constant(-1,shape=[2,3])                # 构造全是常数的矩阵
cc=tf.constant([[1,2],[3,4],[5,6]])
d=tf.range(start=3,limit=15,delta=3)         # 输出数列[3,6,9,12,15]
e=tf.random_normal([2,3],mean=-1,stddev=4)   # [2,3]是shape
shuff=tf.random_shuffle(cc)                  # 对数据进行洗牌，把矩阵的行打乱
sess=tf.Session()
print(sess.run(shuff))

aa=np.zeros((3,3))                           # 将numpy数据转换为tensor
ta=tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7],input2:[2]}))


