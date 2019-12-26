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
# 获取训练数据
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

# 加载训练数据
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

# 采用5折交叉验证
kf = KFold(n_splits=5)
id_,text_id,X1,X2,Y=[],[],[],[],[]
f1_news_score = []
recall_news_score = []
precision_news_score = []
iteration_count = 1

result_report = []
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
# 采用五折交叉验证，获取数据来进行训练
for train,test in kf.split(range(len(train_data[1:61]))):
    save_path = 'model/model'
    save_path += str(iteration_count)
    filepath = "model_{epoch:02d}-{val_crf_viterbi_accuracy:.4f}.hdf5"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        keras.callbacks.ModelCheckpoint(os.path.join(save_path, filepath),
                                        monitor='val_loss', save_best_only=True, verbose=0),
    ]
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
    pred_label_list = []  #代表预测标签系列
    true_label_list = []   #代表真实标签序列
    tid_list = []    #代表文章id序列, 加入结果集
    text_list = []    #代表正文序列, 加入结果集
    type_list = []    #代表类型，Manuslly代表手动标注的结果，Predict代表模型预测的结果, 加入结果集
    tag_str_list = []  #代表预测序列, 加入结果集
    car_list = []  #代表汽车序列, 加入结果集
    company_list = [] #代表公司序列, 加入结果集
    person_list = [] #代表人名序列, 加入结果集
    technology_list = []    # 代表技术序列, 加入结果集
    hardware_list = []    # 代表硬件序列, 加入结果集
    software_list = []    # 代表软件序列, 加入结果集
    patent_list = []    # 代表专利序列, 加入结果集
    date_list = []    # 代表日期序列, 加入结果集
    loc_list = []     # 代表地点序列, 加入结果集
    service_list = []  # 代表服务序列, 加入结果集
    for i in range(len(id_test)):
        sample_X = X1_test[i]  #sample_X代表所获得的文本
        sample_text_id = text_id_test[i][0]
        print(sample_text_id)
        tid_list.append(sample_text_id)
        tid_list.append(sample_text_id)
        sample_data = train_data[sample_text_id]
        text_list.append(train_data[sample_text_id][1])
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
        tag_str_list.append(true_str)
        tag_str_list.append(pred_str)
        type_list.append('Manually')
        type_list.append('Predict')
        X_data = [c for c in sample_data[1]]

        # 预测汽车
        car_eneity_pre = get_entity(X_data,pred_label,'B-CAR','I-CAR')
        car_eneity_true = get_entity(X_data,true_label,'B-CAR','I-CAR')
        car_list.append(car_eneity_true)
        car_list.append(car_eneity_pre)

        # 预测公司
        company_eneity_pre = get_entity(X_data,pred_label,'B-COMPANY','I-COMPANY')
        company_eneity_true = get_entity(X_data,true_label,'B-COMPANY','I-COMPANY')
        company_list.append(company_eneity_true)
        company_list.append(company_eneity_pre)

        # 预测人名
        person_eneity_pre = get_entity(X_data,pred_label,'B-PER','I-PER')
        person_eneity_true = get_entity(X_data,true_label,'B-PER','I-PER')
        person_list.append(person_eneity_true)
        person_list.append(person_eneity_pre)

        # 预测技术
        technology_eneity_pre = get_entity(X_data,pred_label,'B-TEL','I-TEL')
        technology_eneity_true = get_entity(X_data,true_label,'B-TEL','I-TEL')
        technology_list.append(technology_eneity_true)
        technology_list.append(technology_eneity_pre)

        # 预测硬件
        hardware_eneity_pre = get_entity(X_data,pred_label,'B-HARDWARE','I-HARDWARE')
        hardware_eneity_true = get_entity(X_data,true_label,'B-HARDWARE','I-HARDWARE')
        hardware_list.append(hardware_eneity_true)
        hardware_list.append(hardware_eneity_pre)

        # 预测软件
        software_eneity_pre = get_entity(X_data, pred_label, 'B-SOFTWARE', 'I-SOFTWARE')
        software_eneity_true = get_entity(X_data, true_label, 'B-SOFTWARE', 'I-SOFTWARE')
        software_list.append(software_eneity_true)
        software_list.append(software_eneity_pre)

        # 预测服务
        service_eneity_pre = get_entity(X_data, pred_label, 'B-SERVICE', 'I-SERVICE')
        service_eneity_true = get_entity(X_data, true_label, 'B-SERVICE', 'I-SERVICE')
        service_list.append(service_eneity_true)
        service_list.append(service_eneity_pre)

        # 预测专利
        patent_eneity_pre = get_entity(X_data, pred_label, 'B-PATENT', 'I-PATENT')
        patent_eneity_true = get_entity(X_data, true_label, 'B-PATENT', 'I-PATENT')
        patent_list.append(patent_eneity_true)
        patent_list.append(patent_eneity_pre)

        # 预测时间
        date_eneity_pre = get_entity(X_data, pred_label, 'B-TIME', 'I-TIME')
        date_eneity_true = get_entity(X_data, true_label, 'B-TIME', 'I-TIME')
        date_list.append(date_eneity_true)
        date_list.append(date_eneity_pre)

        # 预测地点
        loc_eneity_pre = get_entity(X_data, pred_label, 'B-LOC', 'I-LOC')
        loc_eneity_true = get_entity(X_data, true_label, 'B-LOC', 'I-LOC')
        loc_list.append(loc_eneity_true)
        loc_list.append(loc_eneity_pre)
    dict_result = {'id': tid_list, 'Text': text_list, 'Type':type_list, 'Company':company_list,
                   'Car':car_list,'Person':person_list,'Technique':technology_list,'Hardware':hardware_list,
                   'Software':software_list,'Patent':patent_list,'Date':date_list,'Loc':loc_list ,'SequenceResult':tag_str_list }
    pd_result = pd.DataFrame(dict_result)
    print(pd_result.head())
    pd_result.to_excel('Test_case/Fold' + str(iteration_count - 1) + 'Result.xls', index = False,encoding='utf-8')

print('平均f1值')
print(np.array(f1_news_score).mean())
print('平均recall')
print(np.array(recall_news_score).mean())
print('平均precision')
print(np.array(precision_news_score).mean())
resultScore_write = open('resultScore.txt','w',encoding='utf-8')
resultScore_write.writelines(result_report)
resultScore_write.close()