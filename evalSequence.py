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

dict_relation = {'0':'investment','1':'finance','2':'holding','3':'cooperation','4':'apply','5':'products',
                 '6':'carry','7':'implement','8':'appoinment','9':'quit','10':'quit','11':'patents'}
filedir = 'Test_case'
filenames = os.listdir(filedir)
predict_str = []
predict_str_2value = []
for line in filenames:
    filepath = filedir+'/'+line
    reader = open(filepath,encoding='utf-8-sig')
    list_txt = reader.readlines()
    for i in list_txt:
        predict_str.append(i)
print(predict_str [0])
print(len(predict_str))
limitValue = 0.5
# 转换函数，predict—_str为要预测的字符串
# limitValue为限定的阙值
def transformFunction(predict_str,limitValue):
    for i in range(len(predict_str)):
        # predict_str[i] = predict_str[i].repalce('\n','')
        str_list = predict_str[i].split(' ')
        value_list = []
        # print(str_list)
        for j in range(len(str_list)):
            if(str_list[j] != '\n' and float(str_list[j]) >= limitValue):
                value_list.append(dict_relation[str(j)])
            else:
                value_list.append('not')
        # value_list = value_list[0:len(value_list)-1]
        predict_str_2value.append(value_list[0:len(value_list)-1])
    return predict_str_2value
predict_str_2value = transformFunction(predict_str,limitValue)
train_data_2value = []
print(predict_str_2value[0:2])
print(len(predict_str_2value))
reader = open('../train_data/sentences_relation.txt',encoding='utf-8-sig')
train_data = reader.readlines()
for i in range(len(train_data)):
    value_list = []
    if (i % 2 != 0):
        str_list = train_data[i][0:len(train_data[i])-1].split(' ')
        for j in range(len(str_list)):
            if(str_list[j] != '\n'):
                if(str_list[j] == '1'):
                    value_list.append(dict_relation[str(j)])
                else:
                    value_list.append('not')
        train_data_2value.append(value_list)
print(len(train_data_2value))
print(train_data_2value[0:2])
f = open('resultScoreTotal.txt','w',encoding='utf-8')
# train_data_2value = np.array(train_data_2value)
# predict_str_2value = np.array(predict_str_2value)
# print(train_data_2value.shape)
# print(train_data_2value)
# print(predict_str_2value.shape)
# print(predict_str_2value)
f.writelines(classification_report(train_data_2value,predict_str_2value))
f.close()
