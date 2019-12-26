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
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

tf.logging.set_verbosity(tf.logging.ERROR)

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

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
print(tag2idx)

import re
def get_train_data():
    train_data = []
    n2id = []
    count = 0
    reader = open('sentences.txt',encoding = 'utf-8-sig')
    list_data = reader.readlines()
    for i,element in enumerate(list_data):
        if i % 2 != 0:
            n2id = []
            tags_str = list_data[i]
            text_str = list_data[i-1]
            text_str = text_str.replace(' ','')
            text_str = text_str.replace('\n','')
            tags_str = tags_str.replace('\n',' ')
            tags_str_list = tags_str.split(' ')
            for j,e in enumerate(tags_str_list):
                # print(e)
                list_temp = []
                if e != '\n' and e != '':
                    list_temp.append(tag2idx[e])
                    n2id.append(list_temp)
            train_data.append((count,text_str,n2id))
            count = count+1
    return train_data

train_data = get_train_data()
print('数据读取完毕')
print(len(train_data[0]))

print(train_data[0])


# 定义模型
x1_in = keras.layers.Input(shape=(None,))
x2_in = keras.layers.Input(shape=(None,))
bert_output = bert_model([x1_in,x2_in])
lstm = keras.layers.Bidirectional(keras.layers.LSTM(units = 128,return_sequences = True))(bert_output )
drop = keras.layers.Dropout(0.4)(lstm)
dense = keras.layers.TimeDistributed(keras.layers.Dense(128,activation='relu'))(drop)
crf = CRF(n_tags)
out = crf(dense)
model = keras.models.Model(inputs=[x1_in,x2_in],outputs=out)
model.compile(loss=crf.loss_function,optimizer='adam',metrics=[crf.accuracy])
model.summary()
iteration_count = 1
save_path = 'model'
filepath="model_{epoch:02d}-{val_crf_viterbi_accuracy:.4f}.hdf5"

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,verbose=0),
    keras.callbacks.ModelCheckpoint(os.path.join(save_path,filepath),
monitor='val_loss',save_best_only=True, verbose= 0),
]

kf = KFold(n_splits=5)
id_,text_id,X1,X2,Y=[],[],[],[],[]
f1_news_score = []
recall_news_score = []
precision_news_score = []
iteration_count = 1

result_report = []

for train,test in kf.split(range(len(train_data[1:61]))):
    print("这是第"+str(iteration_count)+'轮交叉验证')
    print('*'*100)
    print(test)
    print('*'*100)
    iteration_count = iteration_count+1
    id_train, text_id_train, X1_train, X2_train, Y_train = [], [], [], [], []
    id_test, text_id_test, X1_test, X2_test, Y_test = [], [], [], [], []
    maxlen = 512
    # 对训练集进行处理
    id_count = 0
    for i in train:
        d = train_data[i]
        text = d[1][:maxlen]
        y = d[2][:maxlen]
        x1, x2 = tokenizer.encode(first=text)
        X1_train.append(x1)
        X2_train.append(X2)
        Y_train.append(y)
        id_train.append(id_count)
        id_count = id_count+1
        text_id_train.append([d[0]])
    X1_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_train, padding="post", value=0)
    X2_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_train, padding="post", value=0)
    Y_train = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=Y_train, padding="post", value=0)
    Y_train = [keras.preprocessing.utils.to_categorical(i, num_classes=n_tags) for i in Y_train]
    # 对测试集进行处理
    id_count = 0
    for i in test:
        d = train_data[i]
        text = d[1][:maxlen]
        y = d[2][:maxlen]
        x1, x2 = tokenizer.encode(first=text)
        X1_test.append(x1)
        X2_test.append(X2)
        Y_test.append(y)
        id_test.append([id_count])
        id_count = id_count + 1
        text_id_test.append([d[0]])
    X1_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_test, padding="post", value=0)
    X2_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1_test, padding="post", value=0)
    Y_test = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=Y_test, padding="post", value=0)
    Y_test = [keras.preprocessing.utils.to_categorical(i, num_classes=n_tags) for i in Y_test]
    print(len(id_train), len(id_test), len(text_id_train), len(text_id_test), X1_train.shape, X1_test.shape,
          X1_train.shape, X1_test.shape, len(Y_train), len(Y_test))
    # 进行训练

    history = model.fit([X1_train, X2_train], np.array(Y_train), batch_size=64, epochs=20,
                        validation_data=([X1_test, X2_test], np.array(Y_test)), verbose=1, callbacks=callbacks)
    # 显示训练信息
    hist = pd.DataFrame(history.history)
    print(hist.head())
    # 进行预测
    test_pred = model.predict([X1_test, X2_test], verbose=1)
    print(test_pred)
    print(test_pred.shape)
    # 定义结果标签
    idx2tag = {i: w for w, i in tag2idx.items()}
    print('tag2idx:', tag2idx)
    print('idx2tag:', idx2tag)

    # 转换实体的预测标签函数
    def pred2label(pred):
        out = []
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(idx2tag[p_i].replace("-PAD-", "O"))
            out.append(out_i)
        return out

    pred_labels = pred2label(test_pred)
    test_labels = pred2label(Y_test)

    # 查看相应的F1值
    print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
    print("F1-score: {:.1%}".format(precision_score(test_labels, pred_labels)))
    print("F1-score: {:.1%}".format(recall_score(test_labels, pred_labels)))
    f1_news_score.append(f1_score(test_labels, pred_labels))
    precision_news_score.append(precision_score(test_labels, pred_labels))
    recall_news_score.append(recall_score(test_labels, pred_labels))
    # 统计相关信息
    print(classification_report(test_labels, pred_labels))
    result_report.append(classification_report(test_labels, pred_labels))
    # 随机抽样
    sample_id = random.sample(range(len(id_test)), 1)[0]
    sample_X1 = X1_test[sample_id]
    sample_X2 = X2_test[sample_id]
    tid = id_test[sample_id][0]
    sample_text_id = text_id_test[sample_id]
    print(sample_text_id)
    sample_data = train_data[tid]
    print(sample_data)
    sample_Y = Y_test[sample_id]
    print(sample_Y)

    predict = model.predict([sample_X1.reshape([1, -1]), sample_X2.reshape([1, -1])])
    print(predict.shape)

    pred = np.argmax(predict, axis=-1).reshape([-1])
    true = np.argmax(sample_Y, axis=-1)

    pred_label = [idx2tag[i] for i in pred]
    true_label = [idx2tag[i] for i in true]

    for c, t, p in zip(sample_data[1], pred_label, true_label):
        if t != "-PAD-":
            print("{:15}: {:5} {}".format(c, t, p))
    # 统计每轮验证的结果
    pred_label_list = []
    true_label_list = []
    tid_list = []
    text_list = []
    for i in range(len(id_test)):
        sample_X = X1_test[i]  #sample_X代表所获得的文本
        tid = id_test[i][0]   #tid代表测试集的id
        sample_text_id = text_id_test[i][0] #
        tid_list.append(sample_text_id)
        sample_data = train_data[tid]
        text_list.append(train_data[sample_text_id][1])
        sample_Y = Y_test[i]
        predict = model.predict([sample_X.reshape([1, -1]),sample_X.reshape([1, -1])])
        pred = np.argmax(predict, axis=-1).reshape([-1])
        true = np.argmax(sample_Y, axis=-1)
        pred_label = [idx2tag[i] for i in pred]
        pred_str = ''
        for s in pred_label:
            if s != "-PAD-":
                pred_str += s
                pred_str += ' '
        true_label = [idx2tag[i] for i in true]
        true_str = ''
        for s in true_label:
            if s != "-PAD-":
                true_str += s
                true_str += ' '
        pred_label_list.append(pred_str)
        true_label_list.append(true_str)
    dict_result = {'id': tid_list, '文本内容': text_list, '预测标签': pred_label_list, '真实标签': true_label_list}
    pd_result = pd.DataFrame(dict_result)
    print(pd_result.head())
    pd_result.to_excel('Test_case/Fold' + str(iteration_count - 1) + 'Result.xls', encoding='utf-8')

print('平均f1值')
print(np.array(f1_news_score).mean())
print('平均recall')
print(np.array(recall_news_score).mean())
print('平均precision')
print(np.array(precision_news_score).mean())
resultScore_write = open('resultScore.txt','w',encoding='utf-8')
resultScore_write.writelines(result_report)
resultScore_write.close()