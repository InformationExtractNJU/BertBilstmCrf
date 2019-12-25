#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
tf.logging.set_verbosity(tf.logging.ERROR)


# In[2]:


config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'


# In[3]:


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


# In[4]:


tokenizer = OurTokenizer(token_dict)
bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len = None)


# In[5]:


bert_model.outputs


# In[6]:


# 标签统计
tags = ['O', 'B-COMPANY', 'I-COMPANY', 'B-TEL', 'I-TEL', 'B-CAR', 'I-CAR', 'B-HARDWARE', 'I-HARDWARE', 'B-PATENT', 'I-PATENT', 'B-SOFTWARE', 'I-SOFTWARE', 'B-PER', 'I-PER', 'B-SERVICE', 'I-SERVICE', 'B-TIME', 'I-TIME', 'B-LOC', 'I-LOC']
tag2idx = {tag:i+1 for i, tag in enumerate(tags)}
tag2idx['-PAD-'] = 0
n_tags = len(tag2idx)
tag2idx


# In[7]:


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
            tags_str.replace('\n',' ')
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


# In[8]:


train_data = get_train_data()
print('数据读取完毕')
print(len(train_data[0]))


# In[9]:


print(train_data[0])
a = '123456'
print(a[:512])


# In[10]:


id_,text_id,X1,X2,Y=[],[],[],[],[]
maxlen = 512
for i in range(len(train_data)):
    d = train_data[i]
    text = d[1][:maxlen]
    y = d[2][:maxlen]
    x1,x2 = tokenizer.encode(first=text)
    X1.append(x1)
    X2.append(X2)
    Y.append(y)
    id_.append([i])
    text_id.append([d[0]])
X1 = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=X1, padding="post", value=0)
Y = keras.preprocessing.sequence.pad_sequences(maxlen=maxlen, sequences=Y, padding="post", value=0)
Y = [keras.preprocessing.utils.to_categorical(i, num_classes=n_tags) for i in Y]


# In[11]:


len(id_), len(text_id), X1.shape,len(Y), Y[0].shape


# In[12]:


id_train, id_test, text_id_train, text_id_test, X1_train,  X1_test, X2_train,  X2_test, Y_train, Y_test = train_test_split(id_, text_id, X1, X1, Y, test_size=0.1)


# In[13]:


len(id_train), len(id_test), len(text_id_train), len(text_id_test), X1_train.shape, X1_test.shape, X1_train.shape, X1_test.shape,len(Y_train), len(Y_test)


# In[14]:


# 定义模型
x1_in = keras.layers.Input(shape=(None,))
x2_in = keras.layers.Input(shape=(None,))
bert_output = bert_model([x1_in,x2_in])
lstm = keras.layers.Bidirectional(keras.layers.LSTM(units = 128,return_sequences = True))(bert_output)
drop = keras.layers.Dropout(0.4)(lstm)
dense = keras.layers.TimeDistributed(keras.layers.Dense(128,activation='relu'))(drop)
crf = CRF(n_tags)
out = crf(dense)_
model = keras.models.Model(inputs=[x1_in,x2_in],outputs=out)
model.compile(loss=crf.loss_function,optimizer='adam',metrics=[crf.accuracy])
model.summary()


# In[15]:


save_path = 'model'
filepath="model_{epoch:02d}-{val_crf_viterbi_accuracy:.4f}.hdf5"


# In[16]:


callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,verbose=0),
    keras.callbacks.ModelCheckpoint(os.path.join(save_path,filepath),
monitor='val_loss',save_best_only=True, verbose= 0),
]


# In[29]:


history = model.fit([X1_train, X2_train], np.array(Y_train), batch_size=64, epochs=20,
                    validation_data=([X1_test, X2_test], np.array(Y_test)), verbose=1, callbacks=callbacks)


# In[18]:


hist = pd.DataFrame(history.history)
hist.head()


# In[19]:


# In[20]:


test_pred = model.predict([X1_test,X2_test],verbose = 1)
print(test_pred)


# In[21]:


test_pred.shape


# In[22]:

# 定义结果标签
idx2tag = {i: w for w, i in tag2idx.items()}
print('tag2idx:',tag2idx)
print('idx2tag:',idx2tag)


# In[30]:

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


# In[31]:

# 查看相应的F1值
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))


# In[25]:


# 统计相关信息
print(classification_report(test_labels, pred_labels))


# In[32]:


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


# In[33]:


predict = model.predict([sample_X1.reshape([1, -1]), sample_X2.reshape([1, -1])])
print(predict.shape)


# In[34]:


pred = np.argmax(predict, axis=-1).reshape([-1])
true = np.argmax(sample_Y, axis=-1)


# In[35]:


pred_label = [idx2tag[i] for i in pred]
true_label = [idx2tag[i] for i in true]


# In[36]:


for c, t, p in zip(sample_data[1], pred_label, true_label):
    if t != "-PAD-":
        print("{:15}: {:5} {}".format(c, t, p))


# In[ ]:




