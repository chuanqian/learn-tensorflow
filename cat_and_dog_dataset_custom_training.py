# -*- coding: utf-8 -*-
"""
author     : zhangqianchuan
desc       : 猫狗数据集的自定义训练
maintainer : zhangqianchuan
e-mail     : cc9801053118@163.com
version    : 1.0.0
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

print("Tensorflow version: {}".format(tf.__version__))

train_image_path = glob.glob("./dataset/dc/train/*/*.jpg")
print(len(train_image_path))
