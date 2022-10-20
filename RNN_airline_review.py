import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 通过pandas读入数据
data = pd.read_csv("./dataset/Tweets.csv")

# pandas去除不需要的列表数据
data = data[["airline_sentiment","text"]]


data.airline_sentiment.unique()
data.airline_sentiment.value_counts()
# negative    9178
# neutral     3099
# positive    2363
# Name: airline_sentiment, dtype: int64

positive_data = data[data.airline_sentiment == "positive"]
negative_data = data[data.airline_sentiment == "negative"]
negative_data = negative_data.iloc[:len(positive_data)]

data = pd.concat([positive_data,negative_data])
data = data.sample(len(data))
data["review"] = (data.airline_sentiment == "positive").astype("int")

del data["airline_sentiment"]

# tf.keras.layers.Embedding把文本向量化
import re
token = re.compile("[A-Za-z]+|[!?,.()]")

# 提取文本信息
def reg_text(text):
    new_text = token.findall(text)
    new_text = [word.lower() for word in new_text]
    return new_text


# 文本规范化
data["text"] = data.text.apply(reg_text)
# 英文单词每一个对应成一个整数序列
word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word)


maxword = len(word_set) + 1
word_list = list(word_set)
word_list.index("spending")
word_index_dict = dict((word, word_list.index(word)+1) for word in word_list)
data_train_data = data.text.apply(lambda x: [word_index_dict.get(word, 0) for word in x])
maxlen = max(len(x) for x in data_train_data)
data_train_data = keras.preprocessing.sequence.pad_sequences(data_train_data.values, maxlen=maxlen)

# 构建模型
model = keras.Sequential()
# Embeding : 把文本映射为一个密集向量
model.add(layers.Embedding(maxword, 50, input_length=maxlen))
model.add(layers.LSTM(64))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"]
)
history = model.fit(
    data_train_data,
    data.review.values,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

