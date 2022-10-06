# -*- coding: utf-8 -*-
"""
#File    : softmax_many_classification.py
#Author  : 11789
#Time    : 2022/9/15-19:25
#Desc    : softmax多分类代码实现
#Ver     : 1.0.0
"""
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

print(sys.version_info)
for module in np, pd, tf, keras:
    print(module.__name__, module.__version__)

(train_image, train_label), (test_image, test_label) = keras.datasets.fashion_mnist.load_data()
# plt.imshow(train_image[0])
# plt.show()

train_image = train_image / 255
test_image = test_image / 255

# print(train_image.shape)
"""
使用损失函数sparse_categorical_crossentropy
"""
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])
# model.summary()
# # sparse_categorical_crossentropy 单个数值使用的损失函数
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
# model.fit(train_image, train_label, epochs=5)
# model.evaluate(test_image,test_label)


# train_image_onehot = keras.utils.to_categorical(train_image)
# print(train_image_onehot.shape)
train_label_onehot = keras.utils.to_categorical(train_label)
# test_image_onehot = keras.utils.to_categorical(test_image)
test_label_onehot = keras.utils.to_categorical(test_label)
"""
使用损失函数categorical_crossentropy
"""
# # categorical_crossentropy 向量形式使用的损失函数
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])
# model.summary()
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["acc"])
# model.fit(train_image, train_label_onehot, epochs=5)
# # model.evaluate(test_image, test_label_onehot)
# predict = model.predict(test_image)
# print(predict.shape)
# print(predict[0])
# print(np.argmax(predict[0]))
# print(test_label_onehot[0])


"""
利用Dropout抓包的方式来使得test数据上的升高
"""
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["acc"]
)
history = model.fit(
    train_image,
    train_label_onehot,
    epochs=10,
    validation_data=(test_image, test_label_onehot)
)

print(history.history.keys())

plt.plot(history.epoch, history.history.get('acc'), label="acc")
plt.plot(history.epoch, history.history.get('val_acc'), label="val_acc")
plt.legend()
plt.show()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
plt.show()
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()
