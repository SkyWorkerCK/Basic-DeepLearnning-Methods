# coding:utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, models
import os

# [b, word_num, word_vec]
# [word_num, b, word_vec]最常用

# x = tf.range(10)
# x = tf.random.shuffle(x)
# print(x)
# net = layers.Embedding(10, 5)           # 表示最多处理10个单词，每个单词的维度dim为5
# print(net.trainable_variables)


a = tf.random.normal([4, 3, 5])
print(a)
print(a[:, 0, :])
