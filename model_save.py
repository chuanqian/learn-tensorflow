"""
@Name: model_save.py
@Auth: cc980
@Date: 2022/10/1-16:23
@Desc: 保存模型
@Ver : 1.0.0
"""

import tensorflow as tf

print("Tensorflow version: {}".format(tf.__version__))
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image.shape, train_label.shape
plt.imshow(train_image[0])
train_image = train_image / 255
test_image = test_image / 255
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["acc"]
)
model.fit(train_image, train_label, epochs=3)
model.evaluate(test_image, test_label, verbose=1)
"""
保存整个模型
整个模型可以保存到一个文件中，其中包含权重值，模型撇脂乃至优化器配置。
这样，您就可以为模型设置检查点，并稍后从完全相同的状态继续训练，而无需访问原始代码。
在Keras中保存完全可正常使用的的模型非常有用，您可以在TensorFlow.js中加载它们，然后在网络浏览器中训练它们。
Keras使用HDF5标准提供基本的保存格式。
"""
model.save("less_model.h5")
new_model = tf.keras.models.load_model("less_model.h5")
new_model.summary()
new_model.evaluate(test_image, test_label, verbose=0)
