# -*- coding: utf-8 -*-
"""
#File    : logistic_regression.py
#Author  : 11789
#Time    : 2022/9/13-19:34
#Desc    : 逻辑回归代码实现
#Ver     : 1.0.0
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import sys

print(sys.version_info)
for module in pd, tf:
    print(module.__name__, module.__version__)

data = pd.read_csv("./dataset/credit-a.csv")
print(data.head())
data.iloc[:, -1].value_counts()
x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(15,), activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x, y, epochs=100)

print(history.history.keys())
plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('acc'))
plt.show()
