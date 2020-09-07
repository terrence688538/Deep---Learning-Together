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

def get_generator(noise_img,out_dim,is_train=True,alpha=0.01):  # 生成器
    with tf.variable_scope('generator',reuse=(not is_train)):           # reuse是共享变量的参数     变量在‘generator’这个名字下

