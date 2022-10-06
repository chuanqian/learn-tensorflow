# -*- coding: utf-8 -*-
"""
#File    : multiple_layer_perceptron.py
#Author  : 11789
#Time    : 2022/9/13-18:47
#Desc    : 多层感知器代码实现
#Ver     : 1.0.0
"""
import tensorflow as tf
import pandas as pd
# import matplotlib.pyplot as plt
import sys

print(sys.version_info)
for module in pd, tf:
    print(module.__name__, module.__version__)
data = pd.read_csv("./dataset/Advertising.csv")
# print(data)
# plt.scatter(data.TV, data.sales)
# plt.show()
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100)
test = data.iloc[:10, 1:-1]
print(model.predict(test))
test = data.iloc[:10, -1]
print(test)
