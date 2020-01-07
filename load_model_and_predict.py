import os
import codecs
import re
import random
import string
from tqdm import tqdm
import pandas as pd
import numpy as np
from zhon.hanzi import punctuation
from sklearn.model_selection import train_test_split
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_contrib.layers import CRF
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
# from seqeval.metrics import precision_score, recall_score, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

import re

config_path = '../../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../chinese_L-12_H-768_A-12/vocab.txt'

# 自定义tokenizer
token_dict = {}
with codecs.open(dict_path,'r',encoding='utf-8') as f:
    for line in f:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self,text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R

tokenizer = OurTokenizer(token_dict)

# 获取训练数据
def get_train_data():
    train_data = []
    count = 0
    reader = open('../train_data/sentences_relation.txt',encoding = 'utf-8-sig')
    list_data = reader.readlines()
    for i,element in enumerate(list_data):
        if i % 2 != 0:
            tags_str = list_data[i]
            text_str = list_data[i-1]
            text_str = text_str.replace(' ','')
            text_str = text_str.replace('\n','')
            tags_str = tags_str.replace('\n',' ')
            # tags_str_list = tags_str.split(' ')
            train_data.append((count,text_str,tags_str))
            count = count+1
    return train_data

# 加载训练数据
train_data = get_train_data()
print('数据读取完毕')
print(len(train_data[0]))
print(train_data[0])
train_data=train_data


bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len = None)

# 定义模型
x1_in = keras.layers.Input(shape=(None,))
x2_in = keras.layers.Input(shape=(None,))
bert_output = bert_model([x1_in,x2_in])

sen_vector = Lambda(lambda x: x[:, 0])(bert_output) # 取出[CLS]对应的向量用来做分类
out=Dense(12,activation='sigmoid')(sen_vector)

# lstm = keras.layers.Bidirectional(keras.layers.LSTM(units = 128,return_sequences = True))(bert_output )
# drop = keras.layers.Dropout(0.4)(lstm)
# dense = keras.layers.TimeDistributed(keras.layers.Dense(128,activation='relu'))(drop)
# crf = CRF(n_tags)
# out = crf(dense)
model = keras.models.Model(inputs=[x1_in,x2_in],outputs=out)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.load_weights('model/model_01.hdf5')

X1_test, X2_test = [], []
# 对训练集进行处理
maxlen=512
for d in train_data:
    text = d[1][:maxlen]
    y = d[2][:maxlen]
    x1, x2 = tokenizer.encode(first=text)
    X1_test.append(x1)
    X2_test.append(x2)
    # print('#'*100)
    #         # print (y_train_list)

X1_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_test, padding="post", value=0)
X2_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_test, padding="post", value=0)

test_pred = model.predict([X1_test, X2_test], verbose=1)
# print(test_pred)
write_to_txt=[]
for i in range(len(X1_test)):
    # print (type(X1_test[i]))
    # print (type(test_pred[i]))
    # print (X1_test[i])
    # print (test_pred[i])
    # pred_labels = np.array(listpred2label(test_pred[i].tolist()))
    pred_labels = np.array(test_pred[i].tolist())
    list_pred_labels = [str(x) for x in pred_labels]
    str_pred_labels = ' '.join(list_pred_labels)
    print (train_data[i])
    write_to_txt.append(train_data[i][1])
    write_to_txt.append('\n')
    write_to_txt.append(str_pred_labels)
    write_to_txt.append('\n')
output_path="Test_case/output.txt"
print(write_to_txt[i])
resultwrite = open(output_path, 'w', encoding='utf-8')
resultwrite.writelines(write_to_txt)