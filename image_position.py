"""
@Name: image_position.py
@Auth: cc980
@Date: 2022/10/2-19:01
@Desc: 图片定位
@Ver : 1.0.0
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from lxml import etree
import numpy as np
import glob

print("Tensorflow version: {}".format(tf.__version__))
img = tf.io.read_file("./dataset/image_position_dataset/images/Abyssinian_1.jpg")
img = tf.image.decode_jpeg(img)
print(img.shape)
