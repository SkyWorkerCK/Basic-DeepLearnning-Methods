# coding:utf-8

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow import keras
import numpy as np
import os

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 情感分类问题
total_words = 10000
max_review_len = 80                           # 句子长度
embedding_len = 100                           # 每个单词嵌入的长度
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

# 定义BATCHSIZE,一次训练多少个句子
batchsz = 128

# 把句子搞成相同长度便于训练
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# 形成数据集
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

# x_train:[b, 80]
# x_test: [b, 80]


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # 初始化状态 [b, 64]
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]

        # 将句子中的单词进行embedding
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # RNN Cell
        # [b, 80, 100], h_dim = 64(RNN Cell当中的内部维度)
        self.run_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        self.run_cell1 = layers.SimpleRNNCell(units, dropout=0.2)


        # 建立全连接层 [b, 80, 100] => [b, 64] => [b, 1]（二分类问题）
        self.outlay = layers.Dense(1)

    # 前向传播
    def call(self, inputs, training=None):
        # [b, 80]
        x = inputs
        # [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # 设置初始状态
        state0 = self.state0
        state1 = self.state1
        # RNN   word: [b, 100]
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.run_cell0(word, state0, training)
            out1, state1 = self.run_cell1(out0, state1)
            state0 = state1
        # out: [b, 64] => [b, 1]
        x = self.outlay(out1)
        print("x:", x)
        prob = tf.sigmoid(x)
        print("prob:", prob)

        return prob

def main():
    # hide layer
    units = 64
    # 训练轮数
    epochs = 4

    model = MyRNN(units)
    # 定义优化器
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    # 对数据集进行测试
    model.evaluate(db_test)

if __name__ == '__main__':
    main()