"""
@Name: custom_model_save_recover.py
@Auth: cc980
@Date: 2022/10/2-16:32
@Desc: 自定义模型恢复
@Ver : 1.0.0
"""
import tensorflow as tf
import os
import tqdm

# 环境变量的配置
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 数据的加载
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(60000)

# 模型的构建
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

# 模型的相关配置
test_accuracy = tf.keras.metrics.Accuracy('test_accuracy')

# 定义模型保存的函数
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(r'save_check'))


def step_test(mol, imags, labels):
    pre = mol(imags, training=False)
    test_accuracy(labels, tf.argmax(pre, axis=-1))


tqdm_test = tqdm.tqdm(iter(test_dataset), total=len(test_dataset))

for img, lable in tqdm_test:
    step_test(model, img, lable)
    tqdm_test.set_postfix_str(test_accuracy.result())
