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

h_dim = 20              # hide layer dimension
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

# 定义超参数
z_dim = 10

# 创建变分自编码器
class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()


        # Encoder     fc1 => fc2         fc1 => fc3
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)

        # decoder
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        # 获取期望
        mu = self.fc2(h)
        # 获取方差
        log_var = self.fc3(h)

        return mu, log_var

    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    def reparameterize(self, mu, log_var):
        # 从标准正态分布中进行取样
        eps = tf.random.normal(log_var.shape)
        # print("eps.shape:{}".format(eps.shape))
        # 计算标准差
        std = tf.exp(log_var) ** 0.5
        # print("std.shape:{}".format(std.shape))
        # reparameterize trick
        z = mu + std * eps
        # print("z.shape:{}".format(z.shape))

        return z

    def call(self, inputs, training=None):
        # 获得期望和方差
        mu, log_var = self.encoder(inputs)
        # print("mu.shape:{}, log_var.shape:{}".format(mu.shape, log_var.shape))
        # 采样  ————  reparameterize
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

if __name__ == '__main__':
    model = VAE()
    model.build(input_shape=(10, 784))
    optimizer = tf.optimizers.Adam(lr)

    for epoch in range(1000):
        for step, x in enumerate(train_db):
            x = tf.reshape(x, [-1, 784])
            with tf.GradientTape() as tape:
                x_rec_logits, mu, log_var = model(x)
                # 计算损失函数
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
                rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

                # 计算KL散度    p:N(mu, log_var)  ~   q:N(0, 1)
                # 根据KL散度的公式进行化简
                kl_div = -0.5 * (log_var + 1 - mu**2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / x.shape[0]

                loss = rec_loss + 1. * kl_div

            # 计算梯度
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 打印
            if epoch % 100 == 0:
                print("epoch:{}, step：{}，KL_value:{}, rec_loss:{}".format(epoch, step, float(kl_div), rec_loss))

    #     # evaluation
    #     z = tf.random.normal((batchsz, z_dim))
    #     logits = model.decoder(z)
    #     x_hat = tf.sigmoid(logits)
    #     x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    #     x_hat = x_hat.astype(np.uint8)
    #     save_images(x_hat, 'vae_images/sampled_epoch%d.png' % epoch)
    #
    #     x = next(iter(test_db))
    #     x = tf.reshape(x, [-1, 784])
    #     x_hat_logits, _, _ = model(x)
    #     x_hat = tf.sigmoid(x_hat_logits)
    #     x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    #     x_hat = x_hat.astype(np.uint8)
    #     save_images(x_hat, 'vae_images/recons_epoch%d.png' % epoch)
    #
    #
    #
