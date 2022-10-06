# -*- coding: utf-8 -*-
"""
#File    : learn_regression.py
#Author  : 11789
#Time    : 2022/9/12-16:29
#Desc    : 线性回归算法
#Ver     : 1.0.0
"""
# 导入相关包比如：tensorflow，numpy，pandas，matplotlib等
import tensorflow as tf
import pandas as pd

# import matplotlib.pyplot as plt

print('Tensorflow Version: {}'.format(tf.__version__))  # 查看Tensorflow的版本
data = pd.read_csv('./dataset/Income1.csv')  # 输入数据
# print(data)
# plt.scatter(data.Education, data.Income)
# plt.show()
x = data.Education
y = data.Income
model = tf.keras.Sequential()  # 创建一个线性回归模型
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.summary()

model.compile(optimizer='adam', loss='mse')  # 使用梯度下降法，比较函数是mse
history = model.fit(x, y, epochs=5000)  # 训练5000次

print(model.predict(x))  # 预测模型输出
print(model.predict(pd.Series([20])))
