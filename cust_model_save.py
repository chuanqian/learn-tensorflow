"""
@Name: cust_model_save.py
@Auth: cc980
@Date: 2022/10/2-16:19
@Desc: 自定义训练的模型保存
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
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.batch(60000)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(60000)

# 模型的构建
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

# 模型的相关配置
optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss_mean = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.Accuracy('train_accuracy')

test_loss_mean = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.Accuracy('test_accuracy')

# 定义模型保存的函数
checkpoint = tf.train.Checkpoint(model=model)


# 定义单步的训练
def step_train(mol, images, labels):
    with tf.GradientTape() as t:
        pre = mol(images)
        loss_step = loss_func(labels, pre)
    grads = t.gradient(loss_step, mol.trainable_variables)
    optimizer.apply_gradients(zip(grads, mol.trainable_variables))
    train_loss_mean(loss_step)
    train_accuracy(labels, tf.argmax(pre, axis=-1))


def step_test(mol, imags, labels):
    pre = mol(imags, training=False)
    loss_step = loss_func(labels, pre)
    test_loss_mean(loss_step)
    test_accuracy(labels, tf.argmax(pre, axis=-1))


# 定义训练函数
def train():
    for i in range(300):
        tqdm_train = tqdm.tqdm(iter(dataset), total=len(dataset))
        for img, lab in tqdm_train:
            step_train(model, img, lab)
            tqdm_train.set_description_str('Epoch : {:3}'.format(i))
            tqdm_train.set_postfix_str(
                'train loss is {:.14f} train accuracy is {:.14f}'.format(train_loss_mean.result(),
                                                                         train_accuracy.result()))
        tqdm_test = tqdm.tqdm(iter(test_dataset), total=len(test_dataset))
        for ima, lbl in tqdm_test:
            step_test(model, ima, lbl)
            tqdm_test.set_description_str('Epoch : {:3}'.format(i))
            tqdm_test.set_postfix_str(
                'test loss is {:.14f} test accuracy is {:.14f}'.format(test_loss_mean.result(), test_accuracy.result()))
        if i % 50 == 0:
            checkpoint.save(file_prefix=r'save_check/logs')
        train_loss_mean.reset_states()
        train_accuracy.reset_states()
        test_loss_mean.reset_states()
        test_accuracy.reset_states()
        tqdm_train.close()
        tqdm_test.close()


if __name__ == '__main__':
    train()
