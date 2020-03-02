#!/usr/bin/env python
# coding: utf-8
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
from keras.layers import LSTM, Dense
from seqeval.metrics import precision_score, recall_score, f1_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
# from LSTM.NonMasking import *

tf.logging.set_verbosity(tf.logging.ERROR)

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
token_dict

tokenizer = OurTokenizer(token_dict)
# 加载bert模型
bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len = None)

# 标签统计
tags = ['O', 'B-COMPANY', 'I-COMPANY', 'B-TEL', 'I-TEL', 'B-CAR', 'I-CAR', 'B-HARDWARE', 'I-HARDWARE', 'B-PATENT', 'I-PATENT', 'B-SOFTWARE', 'I-SOFTWARE', 'B-PER', 'I-PER', 'B-SERVICE', 'I-SERVICE', 'B-TIME', 'I-TIME', 'B-LOC', 'I-LOC']
tag2idx = {tag:i+1 for i, tag in enumerate(tags)}
tag2idx['-PAD-'] = 0
n_tags = len(tag2idx)
# print(tag2idx)

# 获取训练数据
def get_train_data():
    train_data = []
    count = 0
    reader = open('../train_data/words_relation.txt',encoding = 'utf-8-sig')
    list_data = reader.readlines()
    # list_data = list_data[0:1000]
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

def makeBertCNNModel():
    # 定义模型
    x1_in = keras.layers.Input(shape=(None,))
    x2_in = keras.layers.Input(shape=(None,))
    bert_output = bert_model([x1_in, x2_in])
    bert_output = Lambda(lambda x: x[:, 1:254])(bert_output)  # 取出[CLS]对应的向量用来做分类

    lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(bert_output)
    dropout = keras.layers.Dropout(0.4)(lstm)
    dense1 = keras.layers.Dense(128, activation='relu')(dropout)

    conv1d1 = keras.layers.Conv1D(64, 3, activation='relu')(dense1)
    conv1d2 = keras.layers.Conv1D(64, 3, activation='relu')(conv1d1)
    maxpooling1D = keras.layers.MaxPooling1D(3)(conv1d2)
    convld3 = keras.layers.Conv1D(64, 3, activation='relu')(maxpooling1D)
    convld4 = keras.layers.Conv1D(64, 3, activation='relu')(convld3)
    globalaveragepooling1d = keras.layers.GlobalAveragePooling1D()(convld4)
    out = keras.layers.Dense(12, activation='sigmoid')(globalaveragepooling1d)

    model = keras.models.Model(inputs=[x1_in, x2_in], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# 定义获取实体的函数
def get_entity(X_data, y_data,B_tag,I_tag):
    entity_list = []
    entity_name = ''
    for i,(c,l) in enumerate(zip(X_data,y_data)):
        if l == B_tag:
            entity_name += c
        elif (l == I_tag) and (len(entity_name)) > 0:
            entity_name += c
            if i == len(y_data)-1:
                entity_list.append(entity_name)
        elif l == 'O':
            if(len(entity_name)) > 0:
                entity_list.append(entity_name)
            entity_name = ''
    return ';'.join(list(set(entity_list)))

#转换为0-1的整数
def pred2label(pred):
    out=[]
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            if(p<0.5):
                out_i.append(0)
            else:
                out_i.append(1)
        out.append(out_i)
    return out

#转换为0-1的整数,输入为一维list
def listpred2label(pred):
    out=[]
    for p in pred:
        if(p<0.5):
            out.append(0)
        else:
            out.append(1)
    return out

def trainProcessRelation():
    # 加载训练数据
    train_data = get_train_data()
    print('数据读取完毕')
    print(len(train_data[0]))
    print(train_data[0])

    id_, text_id, X1, X2, Y = [], [], [], [], []

    model=makeBertCNNModel()
    save_path = 'model/model_10'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filepath = "model_{epoch:02d}.hdf5"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        keras.callbacks.ModelCheckpoint(os.path.join(save_path, filepath),
                                        monitor='val_loss', save_best_only=True, verbose=0),
    ]

    id_train, text_id_train, X1_train, X2_train, Y_train = [], [], [], [], []
    id_test, text_id_test, X1_test, X2_test, Y_test = [], [], [], [], []
    maxlen = 256
    # 对训练集进行处理
    id_count = 0
    for i in range(len(train_data)):
        d = train_data[i]
        text = d[1][:maxlen]
        y = d[2][:maxlen]
        x1, x2 = tokenizer.encode(first=text)
        X1_train.append(x1)
        X2_train.append(X2)
        y_train_list = y.split(' ')
        y_train_list = list(filter(None, y_train_list))
        y_train_list = list(map(int, y_train_list))
        Y_train.append(y_train_list)
        id_train.append(id_count)
        id_count = id_count + 1
        text_id_train.append([d[0]])
    X1_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_train, padding="post",
                                                          value=0)
    X2_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_train, padding="post",
                                                          value=0)

    X1_train, X1_test, X2_train, X2_test, Y_train, Y_test= train_test_split(
        X1_train, X2_train, Y_train,test_size=0.1)

    Y_test = np.array(Y_test)
    Y_train = np.array(Y_train)
    # 进行训练
    history = model.fit([X1_train, X2_train], Y_train, batch_size=64, epochs=20,
                        validation_data=([X1_test, X2_test], Y_test), verbose=1, callbacks=callbacks)
    # 显示训练信息
    hist = pd.DataFrame(history.history)
    print(hist.head())
    # 进行预测
    pred = model.predict([X1_test, X2_test], verbose=1)
    pred_labels = pred.tolist()
    test_labels = Y_test.tolist()

    evalSequence(test_labels,pred_labels,0.5)
    # limitValue=0.5
    # pred_labels_list=[]
    # for line in pred_labels:
    #     for i in line:
    #         if(i>limitValue):
    #             pred_labels_list.append(1)
    #         else:
    #             pred_labels_list.append(0)
    #
    # test_labels_list=[]
    # for line in test_labels:
    #     for i in line:
    #         if(i>limitValue):
    #             test_labels_list.append(1)
    #         else:
    #             test_labels_list.append(0)
    #
    # print ("*************************")
    # print (pred_labels_list)
    # print (len(pred_labels_list))
    # print (test_labels_list)
    # print (len(test_labels_list))

    # # 将结果抓换成字符串
    # pred_labels_str = []
    # test_labels_str = []
    # for i in range(len(pred_labels)):
    #     str_temp_pred = ''
    #     str_temp_test = ''
    #     for j in range(len(pred_labels[i])):
    #         str_temp_pred += str(pred_labels[i][j])
    #         str_temp_pred += ' '
    #         str_temp_test += str(test_labels[i][j])
    #         str_temp_test += ' '
    #     pred_labels_str.append(str_temp_pred)
    #     test_labels_str.append(str_temp_test)
    # result_str = ''
    # for i in range(len(pred_labels_str)):
    #     result_str += pred_labels_str[i]
    #     result_str += '\n'


    # write_txt = open('Test_case_bert_cnn/Result.txt', 'w', encoding='utf-8')
    # write_txt.writelines(result_str)
    # write_txt.close()

dict_relation = {'0':'investment','1':'finance','2':'holding','3':'cooperation','4':'apply','5':'products',
                 '6':'carry','7':'implement','8':'appoinment','9':'quit','10':'quit','11':'patents'}

def transformFunction(predict_label_list,limitValue):
    predict_str_2value = []
    for i in predict_label_list:
        value_list = []
        for j in range(len(i)):
            if(float(i[j]) >= limitValue):
                value_list.append(dict_relation[str(j)])
            else:
                value_list.append('not')
        # value_list = value_list[0:len(value_list)-1]
        predict_str_2value.append(value_list[0:len(value_list)-1])
    return predict_str_2value

def evalSequence(test_labels,pred_labels,limitValue):
    predict_str_2value=transformFunction(pred_labels,limitValue)
    train_data_2value=[]
    for i in test_labels:
        value_list = []
        for j in range(len(i)):
            if (int(i[j]) == 1):
                value_list.append(dict_relation[str(j)])
            else:
                value_list.append('not')
        train_data_2value.append(value_list)
    print ("-------------------shape----------------")
    print(train_data_2value)
    print(len(train_data_2value))
    print(predict_str_2value)
    print(len(predict_str_2value))
    f = open('resultScoreTotal_bert_cnn_vaild_on_10.txt', 'w', encoding='utf-8')
    f.writelines(classification_report(train_data_2value, predict_str_2value))
    f.close()

def main():
    trainProcessRelation()

if __name__ == '__main__':
    main()
    print ('now __name__ is %s' % __name__)
