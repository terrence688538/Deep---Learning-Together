import tensorflow as tf
import numpy as np
import pickle                                              # 把结果保存到本地的库
import matplotlib.pyplot as plt
import 深度学习.input_data as input_data

mnist=input_data.read_data_sets('/data')

def get_inputs(real_size,noise_size):          # 真实数据和噪音数据
    real_img=tf.placeholder(tf.float32,[None,real_size])
    noise_img=tf.placeholder(tf.float32,[None,noise_size])
    return real_img,noise_img

def get_generator(noise_img,n_units,out_dim,reuse=False,alpha=0.01):  # 生成器
    with tf.variable_scope('generator',reuse=reuse):           # reuse是共享变量的参数     变量在‘generator’这个名字下
        hidden1=tf.layers.dense(noise_img,n_units)             #noise是该层的输入  n_units是神经元的个数
        #leaky Relu               这里用relu的话会有
        #relu缺点是：Relu的输入值为负的时候，输出始终为0，其一阶导数也始终为0，这样会导致神经元不能更新参数，也就是神经元不学习了，这种现象叫做“Dead Neuron”。
        #为了解决Relu函数这个缺点，在Relu函数的负半区间引入一个泄露（Leaky）值，所以称为Leaky Relu函数
        hidden1=tf.maximum(alpha*hidden1,hidden1)
        #drop out
        hidden1=tf.layers.dropout(hidden1,rate=0.2)
        logits=tf.layers.dense(hidden1,out_dim)                 #创建输出层
        outputs=tf.tanh(logits)                                 #tanh把数据压缩到0-1
        return logits,outputs

def get_discriminator(img,n_unit,reuse=False,alpha=0.01):      #判别器  要使用两次并且共享一组参数
    with tf.variable_scope('discriminator',reuse=reuse):
        hidden1=tf.layers.dense(img,n_unit)
        hidden1=tf.maximum(alpha*hidden1,hidden1)
        logits = tf.layers.dense(hidden1,1)  # 创建输出层
        outputs = tf.sigmoid(logits)  # tanh把数据压缩到0-1
        return logits, outputs

img_size=mnist.train.images[0].shape[0]        #输入大小
noise_size=100                                 #噪声图大小
g_unit=128                                     #生成器隐层参数
d_unit=128                                     #判别器隐层参数
learning_rate=0.001                            #学习率
alpha=0.01

tf.reset_default_graph()
real_img,noise_img=get_inputs(img_size,noise_size)
g_logits,g_outputs=get_generator(noise_img,g_unit,img_size)
d_logits_real,d_outputs_real=get_discriminator(real_img,d_unit)
d_logits_fake,d_outputs_fake=get_discriminator(g_outputs,d_unit,reuse=True)

#判别模型的loss   JS散度
#识别真实图像 真实图像定义标签是1
d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real)))  #传入预测值和真实值
#识别生成图像  生成图像定义标签是0
d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_fake)))  #传入预测值和真实值
#总体loss
d_loss=tf.add(d_logits_real,d_logits_fake)
#生成模型的loss
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake)))

train_vars=tf.trainable_variables()         #得到变量

g_vars=[var for var in train_vars if var.name.startswith('generator')]
d_vars=[var for var in train_vars if var.name.startswith('discriminator')]
#Adam算法的优化器。Adam即Adaptive Moment Estimation（自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正。 随着迭代次数上升学习率下降
d_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)  #通过更新d_vars 优化d_loss
g_train_opt=tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars)  #通过更新g_vars 优化g_loss

batch_size=64
epochs=300
n_sample=25         #可视化的一个参数
samples=[]
losses=[]
saver=tf.train.Saver(var_list=g_vars)    #保存生成器的变量
with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):
            batch=mnist.train.next_batch(batch_size)
            batch_images=batch[0].reshape((batch_size,784))
            #对图像像素进行scale,因为tanh输出的结果介于（-1，1），这里改成（0，1）
            batch_images=batch_images*2-1
            batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_size))

            _=sess.run(d_train_opt,feed_dict={real_img:batch_images,noise_img:batch_noise})
            _=sess.run(g_train_opt,feed_dict={noise_img:batch_noise})
        train_loss_d=sess.run(d_loss,feed_dict={real_img:batch_images,noise_img:batch_noise})
        train_loss_d_real=sess.run(d_loss_real,feed_dict={real_img:batch_images,noise_img:batch_noise})
        train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img: batch_images, noise_img: batch_noise})
        train_loss_g=sess.run(g_loss,feed_dict={noise_img:batch_noise})
        losses.append((train_loss_d,train_loss_d_real,train_loss_d_fake,train_loss_g))
        #保存样本
        sample_noise=np.random.uniform(-1,1,size=(n_sample,noise_size))
        gen_samples=sess.run(get_generator(noise_img,g_unit,img_size,reuse=True),feed_dict={noise_img:sample_noise})
        samples.append(gen_samples)
        saver.save(sess,'./checkpoints/generator.ckpt')
with open('train_samples.pkl','wb') as f:
    pickle.dump(samples,f)

fig,ax=plt.subplots(figsize=(20,7))
losses=np.array(losses)
plt.plot(losses.T[0],label='判别器总损失')
plt.plot(losses.T[1],label='判别真实总损失')
plt.plot(losses.T[2],label='判别生成总损失')
plt.plot(losses.T[3],label='生成器损失')
plt.title('对抗生成网络')
ax.set_xlabel('epoch')
plt.legend()

with open('train_sample.pkl','rb') as f:
    samples=pickle.load(f)

def view_sample(epoch,samples):
    fig,axes=plt.subplots(figsize=(7,7),nrows=5,ncols=5,sharey=True,sharex=True)
    for ax,img in zip(axes.flatten(),samples[epoch][1]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im=ax.imshow(img.reshape((28,28)),cmap='Greys_r')
    return fig,axes

_=view_sample(-1,samples)