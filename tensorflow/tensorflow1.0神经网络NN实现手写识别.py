import tensorflow as tf
import 深度学习.input_data as input_data

minist=input_data.read_data_sets('data/',one_hot=True)                                       #one_hot编码  ej:0-10个数字 0表示为[1,0,0,0,0,0,0,0,0,0,0]   one_hot=true表示当前输入的一个概率值
print('type of minist is %s'%(type(minist)))
print('number of train data is %d'%(minist.train.num_examples))
print('number of test data is %d'%(minist.test.num_examples))

print('数据长什么样')
trainimg=minist.train.images
trainlabel=minist.train.labels
testimg=minist.test.images
testlabel=minist.test.labels
print('type of trainimg is %s'%(type(trainimg)))
print('type of trainlabel is %s'%(type(trainlabel)))
print('type of testimg is %s'%(type(testimg)))
print('type of testlabel is %s'%(type(testlabel)))
print('shape of trainimg is %s'%(trainimg.shape,))
print('shape of trainlabel is %s'%(trainlabel.shape,))
print('shape of testimg is %s'%(testimg.shape,))
print('shape of testlabel is %s'%(testlabel.shape,))

n_input=784
n_hidden_1=256                                                                           #第一层影藏层神经元的个数  关于多少层以及多少神经元一般是根据别人弄好的
n_hidden_2=128                                                                           #第二层影藏层神经元的个数  这里的256和128没有任何意义只是例子
n_classes=10
#占位符操作
x=tf.placeholder('float',[None,n_input])                                                 #float是类型，后边是维度   这里None表示任意一个数，传入任何值都可以 后边784是固定的也就是说每次传入的都是固定的列，神经网络都是这样，因为如果W变了，参数W也就变了
y=tf.placeholder('float',[None,n_classes])
stddev=0.1

#这里是个全连接网络
weights={                                                                                #权值初始化    构造初始化矩阵   mean没有默认是0
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
    'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
         }
biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes])),
}
print('Network ready')

def multilayer_perceptron(_X,_weights,_biases):
    layer_1 =tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['w1']),_biases['b1']))         #nn模块下的激活函数
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    return (tf.matmul(layer_2,_weights['out'])+_biases['out'])

pred=multilayer_perceptron(x,weights,biases)                                           #预测值
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))     #损失函数，不同的问题损失函数不一样
optm=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))                                        #看一下预测概率最大的那个和真实数据最大的是不是index一样
acrr=tf.reduce_mean(tf.cast(corr,'float'))                                             #tf.cast改类型

init=tf.global_variables_initializer()
print('Function ready')

training_epoches=20                                                                    #总共迭代20次    一共50000数据
batch_size=100                                                                         #每次迭代跑100个数据
display_step=4                                                                         #每迭代多少次输出一次结果

sess=tf.Session()
sess.run(init)

for epoch in range(training_epoches):
    avg_cost=0
    total_batch=int(minist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=minist.train.next_batch(batch_size)                           #tf自己的数据集有next_batch，如果是自己的数据集要自己写next_batch
        feeds={x:batch_xs,y:batch_ys}                                                   #我们用了placehold的格式，这里指定一下
        sess.run(optm,feed_dict=feeds)
        avg_cost +=sess.run(cost,feed_dict=feeds)
    avg_cost=avg_cost/total_batch
    if (epoch+1) % display_step==0:
        print('Epoch:%03d/%03d cost:%.9f' %(epoch,training_epoches,avg_cost))
        feeds={x:batch_xs,y:batch_ys}
        train_acc=sess.run(acrr,feed_dict=feeds)
        print('train accuracy:%.3f'%(train_acc))
        feeds={x:minist.test.images,y:minist.test.labels}
        test_acc=sess.run(acrr,feed_dict=feeds)
        print('test accuracy:%.3f'%(test_acc))
print('Finish')
