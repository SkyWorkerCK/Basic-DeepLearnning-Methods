# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 保存图片函数
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

h_dim = 10              # hide layer dimension
batchsz = 512           # 批处理操作量
lr = 1e-3

# 数据集加载，图片为 28 * 28
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 类型转换
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
# 训练集和测试集的划分
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        # Decoder
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    # 前向传播过程
    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat

model = AE()
model.build(input_shape=(None, 784))
model.summary()

# 创建优化器
optimizer = tf.optimizers.Adam(lr=lr)

# 编写训练过程
for epoch in range(100):
    for step, x in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            # 通过AE重建出来的logits
            x_rec_logits = model(x)
            # reconstruction后的损失函数
            # from_logits=True   =>   最后一层的几乎函数是sigmoid
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        # 进行优化
        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 打印
        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

        """
        # 测试效果
        # evaluation
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # [b, 28, 28]  => [2b, 28, 28]
        # x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_hat
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, './ae_images/rec_epoch_%d.png' % epoch)
        """



