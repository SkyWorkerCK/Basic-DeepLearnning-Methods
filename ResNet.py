# coding:utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


# 建立ResNet与DenseNet
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):    # stride=1表示不进行采样  filter_num表示通道数
        # 调用母类的初始化方法
        super(BasicBlock, self).__init__()

        # 定义basicblock的基本结构
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')    # stride=1保持输入输出的参数维度一致
        self.bn2 = layers.BatchNormalization()

        # 定义短接线shortcut
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    # 搭建前向传播过程
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut层
        output = self.downsample(inputs)

        output = layers.add([out, output])
        output = tf.nn.relu(output)

        return output

class ResNet(keras.Model):

    def __init__(self, layer_dims, num_classes=100):           # [2, 2, 2, 2]表示有4个ResBlock,每个ResBlock有2个basicBlock
        super(ResNet, self).__init__()                         # num_classes表示全连接层的输出，有100个类

        # 预处理层
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                            layers.BatchNormalization(),
                            layers.Activation('relu'),
                            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                            ])
        # 创建4个ResBlock
        self.layer1 = self.build_Resblock(64, layer_dims[0])
        self.layer2 = self.build_Resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_Resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_Resblock(512, layer_dims[3], stride=2)

        # 全连接层output=[b, 512, h, w]  layers.GlobalAveragePooling2D()解决h和w未知的情况
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    # 搭建前向传播过程
    def __call__(self, inputs, training=None):
        # 进入预处理层
        x = self.stem(inputs)

        # 进入ResNet层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 进入全连接层   [b, c]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)

        return x



    def build_Resblock(self, filter_num, blocks, stride=1):
        res_block = Sequential()
        res_block.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, stride=1))

        return res_block


def resnet18():
    return ResNet([2, 2, 2, 2])

# def resnet34():
#     return ResNet([3, 4, 6, 3])
