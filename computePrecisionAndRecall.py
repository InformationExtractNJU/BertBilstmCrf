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

threshold=[0.2,0.5,0.3,0.4,0.2,0.5,0.4,0.4,0.2,0.4,0.2,0.5]

#转换为0-1的整数,输入为一维list
def listpred2label(pred):
    # print (pred)
    pred=pred.split(' ')
    out=[]
    for p in range(len(pred)):
        # print ("-------------------------")
        # print(p)
        pred[p]=float(pred[p])
        if(pred[p]<threshold[p]):
            out.append(0)
        else:
            out.append(1)
    return out

reader = open('../train_data/sentences_relation.txt',encoding = 'utf-8-sig')
true_data = reader.readlines()

test_labels=[]
for i in range(len(true_data)):
    if( i%2==1 ):
        str = true_data[i].replace('\n','')
        str = str.split(' ')
        str = list(filter(None, str))
        nums = [int(x) for x in str]
        test_labels.append(nums)

reader = open('Test_case/output.txt',encoding = 'utf-8-sig')
pred_data = reader.readlines()

pred_labels=[]
for i in range(len(pred_data)):
    if (i % 2 == 1):
        temp = listpred2label(pred_data[i])
        pred_labels.append(temp)

test_labels = np.array(test_labels)
pred_labels = np.array(pred_labels)

print (test_labels[0])
print (type(test_labels[0]))
print ("---------------------")
print (pred_labels[0])


print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels, average='micro')))
print("F1-score: {:.1%}".format(precision_score(test_labels, pred_labels, average='micro')))
print("F1-score: {:.1%}".format(recall_score(test_labels, pred_labels, average='micro')))
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels, average='micro')))

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels, average='macro')))
print("F1-score: {:.1%}".format(precision_score(test_labels, pred_labels, average='macro')))
print("F1-score: {:.1%}".format(recall_score(test_labels, pred_labels, average='macro')))
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels, average='macro')))

# f1_news_score.append(f1_score(test_labels, pred_labels, average='micro'))
# precision_news_score.append(precision_score(test_labels, pred_labels, average='micro'))
# recall_news_score.append(recall_score(test_labels, pred_labels, average='micro'))
# 统计相关信息
print(classification_report(test_labels, pred_labels))
# result_report.append(classification_report(test_labels, pred_labels))
# 随机抽样
# # sample_id = random.sample(range(len(id_test)), 1)[0]
# sample_X1 = X1_test[sample_id]
# sample_X2 = X2_test[sample_id]
# tid = id_test[sample_id][0]
# # sample_text_id = text_id_test[sample_id]
# # print(sample_text_id)
# # sample_data = train_data[tid]
# # print(sample_data)
# sample_Y = Y_test[sample_id]
# print(sample_Y)
#
# predict = model.predict([sample_X1.reshape([1, -1]), sample_X2.reshape([1, -1])])
# print(predict.shape)
#
# pred = np.argmax(predict, axis=-1).reshape([-1])
# true = np.argmax(sample_Y, axis=-1)
# print (pred)
# print (true)