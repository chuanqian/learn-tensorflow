import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import sys
import matplotlib as mpl
import tensorflow as tf

# 不同库版本，使用本代码需要查看
print(sys.version_info)
for module in mpl, np, tf, keras:
    print(module.__name__, module.__version__)

'''
sys.version_info(major=3, minor=6, micro=9, releaselevel='final', serial=0)
matplotlib 3.3.4
numpy 1.16.0
tensorflow 1.14.0
tensorflow.python.keras.api._v1.keras 2.2.4-tf
'''
# If you get numpy futurewarning,then try numpy 1.16.0

# Load train and test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape to dense standard input
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.2)
score = model.evaluate(x_test, y_test)

print("Test Loss:", score[0])
print("Test Accuracy", score[1])


# Test Loss: 0.11334700882434845
# Test Accuracy 0.9750999808311462
# 注意天下几乎没有两个相同的准确率，因为你是random出来的，在epoch之前会randomly initialize parameters

# visualize accuracy and loss
def plot_(history, label):
    plt.plot(history.history[label])
    plt.plot(history.history["val_" + label])
    plt.title("model " + label)
    plt.ylabel(label)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


plot_(history, "acc")
plot_(history, "loss")

'''
@Tool : Tensorflow 2.x
Transform acc,val_acc above into accuracy,val_accuracy
'''
