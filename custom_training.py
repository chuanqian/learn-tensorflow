# 自定义训练模型tensorflow
import tensorflow as tf
from tensorflow import keras

print("Tensorflow version: {}".format(tf.__version__))
(train_image, train_label), (test_image, test_label) = keras.datasets.mnist.load_data()
print(train_image.shape)
print(train_label.shape)
# train_image and test_image add dimension
train_image = tf.expand_dims(train_image, -1)
test_image = tf.expand_dims(test_image, -1)
# print(train_image.shape)

# 改变类型并归一化
train_image = tf.cast(train_image / 255, tf.float32)
test_image = tf.cast(test_image / 255, tf.float32)
train_label = tf.cast(train_label, tf.int64)
test_label = tf.cast(test_label, tf.int64)

# print(train_image.shape)
# (60000, 28, 28, 1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label))
BATCH_SIZE = 64
train_dataset.shuffle(10000).batch(BATCH_SIZE).repeat()
test_dataset.batch(BATCH_SIZE)

print(train_dataset.shape)

# 定义模型
# 定义模型，并初始化
model = keras.Sequential([
    keras.layers.Conv2D(16, [3, 3], input_shape=(28, 28, 1), activation="relu"),
    keras.layers.Conv2D(32, [3, 3], activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10)
])
# model.summary()
# 定义优化器
optimizer = keras.optimizers.Adam(learning_rate=0.001)
# 定义模型损失函数调用方法，即可调用
loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# # 定义一个损失函数
# def loss(model, x, y):
#     y_ = model(x)
#     return loss_func(y, y_)


# 定义计算正确值
train_loss = keras.metrics.Mean("train_loss")
train_acc = keras.metrics.SparseCategoricalAccuracy("train_acc")
test_loss = keras.metrics.Mean("test_loss")
test_acc = keras.metrics.SparseCategoricalAccuracy("test_acc")


# 一步训练数据集
def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        # print(pred.shape)
        # print(labels.shape)
        # pred = tf.argmax()
        loss_step = loss_func(labels, pred)
    grads = t.gradient(loss_step, model.trainabel_variables)
    optimizer.apply_gradients(zip(grads, model.trainabel_variables))
    train_loss(loss_step)
    train_acc(labels, pred)


def test_step(model, images, labels):
    pred = model(images)
    loss_step = loss_func(labels, pred)
    test_loss(loss_step)
    test_acc(labels, pred)


def train():
    for epoch in range(10):
        for (batch, (images, labels)) in enumerate(train_dataset):
            train_step(model, images, labels)
        print("Epoch: {}, loss: {}, acc: {}".format(epoch, train_loss.result(), train_acc.result()))
        for (batch, (images, labels)) in enumerate(test_dataset):
            test_step(model, images, labels)
        print("Epoch: {}, loss: {}, acc: {}".format(epoch, test_loss.result(), test_acc.result()))


train()
