"""
@Name: tensorboard_visualization.py
@Auth: cc980
@Date: 2022/9/25-14:30
@Desc: Tensorboard的可视化
@Ver : 1.0.0
"""
import keras
import tensorflow as tf
import datetime
import os

print("Tensorflow version: {}".format(tf.__version__))
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
print(train_image.shape)
train_image = tf.expand_dims(train_image, -1)
test_image = tf.expand_dims(test_image, -1)

train_image = tf.cast(train_image / 255, tf.float32)
test_image = tf.cast(test_image / 255, tf.float32)
train_label = tf.cast(train_label, tf.int64)
test_label = tf.cast(test_label, tf.int64)

dataset_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))
dataset_test = tf.data.Dataset.from_tensor_slices((test_image, test_label))

print(dataset_train)

dataset_train = dataset_train.shuffle(10000).repeat().batch(64)
dataset_test = dataset_test.repeat().batch(64)

model = keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], input_shape=(28, 28, 1), activation="relu"),
    tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["acc"]
)

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
model.fit(
    dataset_train,
    epochs=5,
    steps_per_epoch=60000//64,
    validation_data=dataset_test,
    validation_steps=10000//64,
    callbacks=[tensorboard_callback]
)
