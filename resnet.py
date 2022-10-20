import tensorflow as tf
from tensorflow import keras


class BasicBlock(keras.layers.Layer):
    def __init__(self, fliter_num, strite):
        super(BasicBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(fliter_num, kernel_size=(3, 3), strite=strite, padding="same")
        self.bn1 = keras.layers.BatchNormalization
        self.relu1 = keras.layers.Activation("relu")

        self.conv2 = keras.layers.Conv2D(fliter_num, kernel_size=(3, 3), strite=strite, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.Activation("relu")

        if strite % 2 != 0:
            self.downsample = lambda x: x
        else:
            self.downsample = keras.Sequential()
            self.downsample.add(
                keras.layers.Conv2D(fliter_num, (1, 1), strite=strite, padding="same")
            )

    # input[n, h, w, c]
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = keras.layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


# 创建一个Resnet网络模型
class Resnet(keras.Model):
    def build_resblock(self, fliter_num, blocks, strite=1):
        resblock = keras.Sequential()
        resblock.add(
            BasicBlock(fliter_num, strite)
        )
        for _ in range(1, blocks):
            resblock.add(BasicBlock(fliter_num, strite))
        return resblock

    def __init__(self, layer_dim, num_classes):
        self.stm = keras.Sequential([
            keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding="same")
        ])
        self.layer1 = self.build_resblock(64, layer_dim[0])
        self.layer2 = self.build_resblock(128, layer_dim[1], strite=2)
        self.layer3 = self.build_resblock(256, layer_dim[2], strite=2)
        self.layer4 = self.build_resblock(512, layer_dim[3], strite=2)
        self.fc = keras.layers.Dense(num_classes)
        self.avgpool = keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None):
        x = self.stm(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def myrest(self):
        return Resnet([2, 2, 2, 2], 100)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def load_data():
    (x, y), (x_test, y_test) = keras.datasets.cifar100.load_data()
    y = tf.squeeze(y)
    y_test = tf.squeeze(y)
    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.map(preprocess).shuffle(50000).batch(64)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).batch(64)
    return db_train, db_test


(db_train, db_test) = load_data()


def main():
    network = Resnet.myrest()
    network.build_resblock(input_shape=(None, 32, 32, 3))
    network.summary()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    for epoch in range(20):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                pred = network(x)
                y_onehot = tf.one_hot(y, 100)
                loss = tf.losses.categorical_crossentropy(y_onehot, pred, from_logits=True)
                loss = tf.reduce_mean(loss)
            greds = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(greds, network.trainable_variables))
            if step % 100 == 0:
                print("loss: ,", float(loss))
    total_number = 0
    correct_number = 0
    for step, (x, y) in enumerate(db_test):
        out = network(x)
        y = tf.cast(y, dtype=tf.int32)
        prod = tf.nn.softmax(out, axis=1)
        pred = tf.argmax(prod, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
        correct_number += correct
        total_number += x.shape[0]
    print("acc: ", float(correct_number / total_number))

if __name__ == '__main__':
    main()
