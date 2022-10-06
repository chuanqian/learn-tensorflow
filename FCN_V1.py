import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
%matplotlib inline
import glob

images = glob.glob(r"./dataset/image_position_dataset/images/*.jpg")

# 读取目标图像
label_images = glob.glob(r"./dataset/image_position_dataset/annotations/trimaps/*.png")


# 排序，保证顺序一致
images.sort(key=lambda x:x.split("\\")[-1].split(".jpg")[0])
label_images.sort(key=lambda x:x.split("\\")[-1].split(".png")[0])


# 做乱序处理，但images和label_images也是一样的
np.random.seed(2019)
index = np.random.permutation(len(images))
images = np.array(images)[index]
label_images = np.array(label_images)[index]

dataset = tf.data.Dataset.from_tensor_slices((images, label_images))
test_count = int(len(images)*0.2)
train_count = len(images)-test_count
train_dataset = dataset.skip(test_count)
test_dataset = dataset.take(test_count)

def read_jpg(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def read_png(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    return image


def normal_image(input_image, input_label_image):
    input_image = tf.cast(input_image, tf.float32)
    input_image = input_image/127.5 - 1
    input_label_image = input_label_image - 1
    return input_image, input_label_image

@tf.function
def load_images(input_image_path, input_label_image_path):
    input_image = read_jpg(input_image_path)
    input_image = tf.image.resize(input_image, [224,224])
    input_label_image = read_png(input_label_image_path)
    input_label_image = tf.image.resize(input_label_image, [224,224])
    return normal_image(input_image,input_label_image)


train_dataset = train_dataset.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

BATCH_SIZE = 8
# 乱序和设置每次取多少的问题
train_dataset = train_dataset.repeat().shuffle(100).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

for image, label_image in train_dataset.take(1):
    plt.subplot(1,2,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
    plt.subplot(1,2,2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(label_image[0]))

# 使用预训练网络
# 使用VGG16与训练网络
conv_base = tf.keras.applications.VGG16(
    weights="imagenet",
    input_shape=(224,224,3),
    include_top=False
)
conv_base.summary()

# 拿到最后一层的输出做上采样，上采样：反卷积
# 7*7*512，反卷积：14*14*512
# 获取预训练网络的中间层: 切片，也可以是使用get_layer方法
# 如何创建一个子的model： 函数式API
# 创建的子模型，会继承VGG16的预训练的权重
sub_model = tf.keras.models.Model(
    inputs=conv_base.input,
    outputs=conv_base.get_layer('block5_conv3').output
)
sub_model.summary()

# 如何一次性获得多输出层
# 列表推导式
layer_names = [
    "block5_conv3",
    "block4_conv3",
    "block3_conv3",
    "block5_pool"
]
layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]
multi_out_model = tf.keras.models.Model(
    inputs=conv_base.input,
    outputs=layers_output
)
# 不允许训练
multi_out_model.trainable = False


# FCN模型创建
# 根据语义分割的模型创建自己需要的语义分割模型
# 模型输入
inputs = tf.keras.layers.Input(shape=(224,224,3))
output_block5_conv3,output_block4_conv3,output_block3_conv3,output_block5_pool = multi_out_model(inputs)
# 上采样
reverse_one = tf.keras.layers.Conv2DTranspose(
    512,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(output_block5_pool)
# 增加一层卷积提取特征
reverse_one = tf.keras.layers.Conv2D(512,3,padding="same",activation="relu")(reverse_one)
reverse_two = tf.add(reverse_one,output_block5_conv3)
reverse_two = tf.keras.layers.Conv2DTranspose(
    512,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(reverse_two)
reverse_two = tf.keras.layers.Conv2D(512,3,padding="same",activation="relu")(reverse_two)
reverse_three = tf.add(reverse_two,output_block4_conv3)
reverse_three = tf.keras.layers.Conv2DTranspose(
    256,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(reverse_three)
reverse_three = tf.keras.layers.Conv2D(256,3,padding="same",activation="relu")(reverse_three)
reverse_four = tf.add(reverse_three,output_block3_conv3)
reverse_four = tf.keras.layers.Conv2DTranspose(
    128,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(reverse_four)
reverse_four = tf.keras.layers.Conv2D(128,3,padding="same",activation="relu")(reverse_four)
prediction = tf.keras.layers.Conv2DTranspose(
    3,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(reverse_four)
model = tf.keras.models.Model(
    inputs=inputs,
    outputs=prediction
)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["acc"]
)
history = model.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=train_count//BATCH_SIZE,
    validation_data=test_dataset,
    validation_steps=test_count//BATCH_SIZE
)
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(5)
plt.figure()
plt.plot(epochs, loss, "r", label="Training loss")
plt.plot(epochs, loss, "r", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.ylim([0,1])
plt.legend()
plt.show()
for image, mask in test_dataset.take(1):
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    
    plt.figure(figsize=(10,10))
    for i in range(num):
        plt.subplot(num, 3, i*num+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))
        plt.subplot(num, 3, i*num+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))
        plt.subplot(num, 3, i*num+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))