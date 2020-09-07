import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

LEARNING_RATE=1e-4
TRAINING_ITERATIONS=2500

DROPOUT=0.5
BATCH_SIZE=50

VALIDATION_SIZE=2000
IMAGE_TO_DISPLAY=10

data=pd.read_csv(r'C:\Users\Mechrevo\Desktop\源码与课件\mnist\mnist_dataset\mnist_train.csv')
print('data({0[0]},{0[1]})'.format(data.shape))
print(data.head())

images=data.iloc[:,1:].values                   #values可以把列名去掉
images=images.astype(np.float)                  #astype()函数可用于转化dateframe某一列的数据类型
images=np.multiply(images,1/255)                #归一化处理

image_size=images.shape[1]
image_width=image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)

def display(img):
    one_image=img.reshape(image_width,image_height)
    plt.axis('off')
    plt.imshow(one_image,cmap=cm.binary)

display(images[IMAGE_TO_DISPLAY])

labels_flat=data['5'].values.ravel()
labels_count=np.unique(labels_flat).shape[0]
def dense_to_one_hot(labels_dense,num_classes):
    num_labels=labels_dense.shape[0]
    index_offset=np.arange(num_labels)*num_classes
    label_one_hot=np.zeros((num_labels,num_classes))
    label_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return label_one_hot
labels=dense_to_one_hot(labels_flat,labels_count)
labels=labels.astype(np.uint8)

#split data into training & validation_size  划分验证和训练       这里是为了简化操作
validation_images=images[:VALIDATION_SIZE]
validation_labels=labels[:VALIDATION_SIZE]
train_images=images[VALIDATION_SIZE:]
train_labels=labels[VALIDATION_SIZE:]
#以上都是数据处理的过程

#权值初始化
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)                  #产生截断正态分布随机数:就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')           #x是数据的输入，W是filter   stride的画中间的两个是我们关注的步长在h和w上滑动一般我们的filter是正方形所以一样
                                                                        #padding=same 就是要做padding
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')   #ksize就是说要在多大的区域上选择

x=tf.placeholder('float',shape=[None,image_size])
y_=tf.placeholder('float',shape=[None,labels_count])

W_conv1=weight_variable([5,5,1,32])                #5,5是filter的大小       1是channel因为这里是灰度图   32是说有32个filter
b_conv1=bias_variable([32])                        #总共32个偏置像

image=tf.reshape(x,[-1,image_width,image_height,1])  #(40000,784)变成（40000，28，28，1）
h_conv1=tf.nn.relu(conv2d(image,W_conv1)+b_conv1)    #变成4000，28，28，32
h_pool1=max_pool_2x2(h_conv1)                        #变成4000，14，14，32

#第二个卷积层
W_conv2=weight_variable([5,5,32,64])                #5,5是filter的大小  前边有32个图，我希望可以得到64个图
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)    #变成4000，14,14,64
h_pool2=max_pool_2x2(h_conv2)                          #变成4000，7,7 ,64


#全连接层
W_fc1=weight_variable([7*7*64,1024])                     #输出是希望是1024
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])            #40000，7，7，64变成40000，3136   -1是个未知量这里起到flatten的作用
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)      #40000*1024

#dropout                             防止过拟合的
keep_prob=tf.placeholder('float')
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)          #keep_prob是保留率

W_fc2=weight_variable([1024,labels_count])
b_fc2=bias_variable([labels_count])

y=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)      #预测值，属于某个数的概率

cross_entropy=-tf.reduce_sum(y_*tf.log(y))             #损失函数
train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

epochs_complete=0
index_in_epoch=0
num_examples=train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_complete
    start=index_in_epoch
    index_in_epoch+=batch_size
    if index_in_epoch>num_examples:
        epochs_complete+=1
        perm=np.arange(num_examples)
        np.random.shuffle(perm)
        train_images=train_images[perm]
        train_labels=train_labels[perm]
        start=0
        index_in_epoch=batch_size
        assert batch_size<=num_examples
    end=index_in_epoch
    return train_images[start:end],train_labels[start:end]

init=tf.global_variables_initializer()
sess=tf.InteractiveSession() #它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的
sess.run(init)

train_accuracies=[]
validation_accuracies=[]
x_range=[]
display_step=1

for i in range(TRAINING_ITERATIONS):
    batch_xs,batch_ys=next_batch(BATCH_SIZE)
    if i%display_step==0 or (i+1)==TRAINING_ITERATIONS:          #每个多少次展示结果
        train_accuracy=accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1})    #eval是去掉最外层的引号
        if(VALIDATION_SIZE):
            validation_accuracy=accuracy.eval(feed_dict={x:validation_images[0:BATCH_SIZE],y_:validation_labels[0:BATCH_SIZE],keep_prob:1})
            validation_accuracies.append(validation_accuracy)
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        if i%(display_step*10)==0 and i:
            display_step*=10
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:DROPOUT})










