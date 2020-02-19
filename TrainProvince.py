# coding:utf-8

import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 6
iterations = 300

SAVER_DIR = "./train-saver/province/"
PROVINCES = ("京","闽","粤","苏","沪","浙")
nProvinceIndex = 0
time_begin = time.time()

# 定义输入的数据
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])


# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)


# 主函数入口
if __name__=='__main__':

    # 第一次遍历图片目录是为了获取图片总数
    # 因为一共只有6个省份的数据，所以遍历省份所有训练图片的张数
    inputs_count = 0
    for i in range(0, NUM_CLASSES):
        dir = './tf_car_license_dataset/train_images/training-set/chinese-characters/%s/' % i
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                inputs_count += 1

    # 图片一共有inputs_count张，每张图片有1280个pixel，初始化数组
    input_images = np.array([[0]*SIZE for i in range(inputs_count)])
    value_images = np.array([[0]*SIZE for i in range(inputs_count)])
    # 再初始化标签labels
    input_labels = np.array([[0]*NUM_CLASSES for i in range(inputs_count)])
    value_labels = np.array([[0]*NUM_CLASSES for i in range(inputs_count)])
    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(0, NUM_CLASSES):
        dir = './tf_car_license_dataset/train_images/training-set/chinese-characters/%s/' % i
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w+h*width] = 0
                            value_images[index][w+h*width] = 0
                        else:
                            input_images[index][w+h*width] = 1
                            value_images[index][w + h * width] = 1
                input_labels[index][i] = 1
                value_labels[index][i] = 1
                index += 1

    with tf.Session() as sess:
        # 第一个卷积层
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

        # sess.run(tf.global_variables_initializer())
        # print(sess.run(b_conv1))

        # 定义第二层卷积层
        W_conv2 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv2")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv2")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')


        # 全连接层
        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32 * 512], stddev=0.1), name='W_fc1')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name='b_fc1')
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16*20*30])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)


        # DropOut
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


        # readout层
        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name='W_fc2')
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='b_fc2')


        # 定义优化器和训练op
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)


        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 初始化saver
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        time_elapse = time.time() - time_begin
        print("读取图片文件耗费时间：%d秒" % time_elapse)
        time_begin = time.time()

        print("一共读取了 %s 个训练图像， %s 个标签" % (inputs_count, inputs_count))

        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 60
        iterations = iterations
        batches_count = int(inputs_count / batch_size)
        remainder = inputs_count % batch_size
        print("训练数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))


        # 执行训练迭代
        for it in range(iterations):
            # 这里的关键是要把输入数组转为np.array
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n*batch_size:(n+1)*batch_size], y_: input_labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})
            if remainder > 0:
                start_index = batches_count * batch_size
                train_step.run(feed_dict={x: input_images[start_index:inputs_count-1], y_: input_labels[start_index: inputs_count-1], keep_prob: 0.5})

            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
            iterate_accuracy = 0
            if it%5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: value_images, y_: value_labels, keep_prob: 1.0})
                print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
                if iterate_accuracy >= 0.9999 and it >= iterations:
                    break

        print('完成训练!')
        time_elapsed = time.time() - time_begin
        print("训练耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()

        # 保存训练结果
        if not os.path.exists(SAVER_DIR):
            print('不存在训练数据保存目录，现在创建保存目录')
            os.makedirs(SAVER_DIR)
        # 初始化saver
        saver = tf.train.Saver()
        saver_path = saver.save(sess, "%smodel.ckpt" % (SAVER_DIR))





