# coding:utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
import numpy as np
from WGAN import Generator, Discriminator
from dataset import make_anime_dataset
import os
from scipy.misc import toimage
import glob

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)

def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def gradient_penalty(discriminator, batch_x, fake_image):

    batchsz = batch_x.shape[0]

    # [b, h, w ,c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, 1, 1, 1]  =>  [b, h, w, c]
    tf.broadcast_to(t, batch_x.shape)

    # 差值
    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        # interplate是一个tensor类型，需要加入tape.watch()中
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate)
    grads = tape.gradient(d_interplate_logits, interplate)

    # grads: [b, h, w, c]  =>  [b, -1]
    # 惩罚因子
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp-1)**2)

    return gp


def d_loss_fn(genertor, discriminator, batch_z, batch_x, is_training):
    # train real image as real
    # train fake image as fake
    fake_image = genertor(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    # batch_x 真图片
    # fake_image 假图片
    gp = gradient_penalty(discriminator, batch_x, fake_image)
    # 1. 为超参数
    loss = d_loss_fake + d_loss_real + 1. * gp

    return loss

def g_loss_fn(genertor, discriminator, batch_z, is_training):
    fake_image = genertor(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)

    loss = celoss_ones(d_fake_logits)

    return loss

def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # 设置超参数
    z_dim = 10
    epoch = 3000000
    batch_size = 512
    learning_rate = 0.002
    is_training = True

    # 数据集加载,每一张图片路径集
    img_path = glob.glob(r'D:\PyCharm Projects\CarNum-CNN\data\faces\*.jpg')

    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    # print(dataset, img_shape)
    # sample = next(iter(dataset))
    # print(sample)

    # 无线采样
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    # 导入生成器模型和判断器模型
    genertor = Generator()
    genertor.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    # 分别设置两个优化器
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epoch):

        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)

        # train D
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(genertor, discriminator, batch_z, batch_x, is_training)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # train G
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(genertor, discriminator, batch_z, is_training)

        grads = tape.gradient(g_loss, genertor.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, genertor.trainable_variables))

        # 打印
        if epoch % 100 == 0:
            print(epoch, "d_loss:", float(d_loss), "g_loss:", float(g_loss))

            z = tf.random.uniform([100, z_dim])
            fake_image = genertor(z, training=False)
            img_path = os.path.join('./wgan_images', 'wgan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

if __name__ == '__main__':
    main()



