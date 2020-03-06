import os
import codecs
import pandas as pd


entity_type = ['Company', 'Car', 'Person', 'Technique', 'Hardware', 'Software', 'Patent', 'Date', 'Loc']

entity_to_num = {'Company':0, 'Car':1, 'Person':2, 'Technique':3, 'Hardware':4, 'Software':5, 'Patent':6, 'Date':7, 'Loc':8}
num_to_relation = {0:'investment', 1:'finance', 2:'holding', 3:'cooperation', 4: 'apply', 5: 'products', 6:'carry', 7:'implement',
                 8:'appoinment', 9:'son', 10:'quit', 11:'patents'}

# # 按照描述中出现的顺序定义关系
relation_def = {'investment':[('Person','Company')],
                'finance': [('Company','Company')],
                'holding': [('Company','Company')],
                'cooperation': [('Company','Company')],
                'apply': [('Company','Technique')],
                'products': [('Company','Car'), ('Company','Hardware'), ('Company','Software'), ('Company','Technique')],
                'carry': [('Car','Hardware'), ('Car','Software'), ('Car','Technique')],
                'implement': [('Software','Technique'), ('Hardware','Technique')],
                'appoinment': [('Person','Company')],
                'son': [('Company','Company')],
                'quit': [('Person','Company')],
                'patents': [('Company','Patent')]}

# 从实体抽取结果中读取每一个句子的实体序列，每一个实体序列使用dict存储
def extract_entity(entity_result_file):
    '''
    以下处理datafram获取每一个句子的实体序列，将时间和地点也按照出现顺序放到序列中。
    '''
    entity_trace_list = []
    entity_result_data = pd.read_excel(entity_result_file)
    nrows=entity_result_data.shape[0]
    # print (nrows)
    for i in range(2000):
        # 每隔一行才是预测出来的结果
        if (i % 2==0):
            sentence = entity_result_data.iloc[i][1]
            # print (sentence)
            one_sentences_list = []
            for j in range(len(entity_type)):
                one_type_entity_list = str(entity_result_data.iloc[i][j+3]).split(";")
                one_type_entity_and_index_list=[]
                for e in one_type_entity_list:
                    if(e=='nan'):
                        e_index = -1
                    else:
                        try:
                            e_index = sentence.index(e)
                        except:
                            e_index = -1
                    one_type_entity_and_index_list.append([e,e_index])
                # 记录下实体的时候同时记录下实体在句子中出现的位置
                one_sentences_list.append(one_type_entity_and_index_list)
            entity_trace_list.append(one_sentences_list)
    return entity_trace_list


# 从关系结果中获取每一个句子的关系预测结果，每一个关系序列使用dict存储
def extract_relation(relation_result_file):
    relations_list = []
    reader = open(relation_result_file, encoding = 'utf-8-sig')
    relation_lines = reader.readlines()
    for line in relation_lines:
        relations = dict()
        for item in line.strip().split('\t'):
            pair = item.split(':')
            relations[pair[0]] = float(pair[1])
        relations_list.append(relations)
    return relations_list


# 读取实体预测标签结果序列，构造实体序列；从实体结果中读取时间和地点实体，并确定时间和地点在实体序列中的位置。
# 读取关系预测结果，记录关系序列。
# 按照如下逻辑确定关系：
#   1. 遍历相邻实体，查询relation_def中可能存在的关系
#   2. 如果潜在关系为空，则不记录；如果不为空，查看概率，若大于threshold，则记录关系；
def generate_entity_relation(entity_list,relation_list):
        # 三元组结果文件，每一行写入一个句子的所有三元组，格式为'(实体类型:实体内容,关系类型,实体类型:实体内容)\t(...)\t...'
        all_pair_list=[]
        for i in range(len(relation_list)):
            one_sentence_pair_list = []
            for rel in relation_list[i]:
                # print (i)
                relation_name = num_to_relation[rel]
                relation_pair_list = relation_def[relation_name]
                # print (relation_pair_list)
                for rpl in relation_pair_list:
                    # 实体a的类别
                    entity_a = rpl[0]
                    # 实体b的类别
                    entity_b = rpl[1]
                    entity_list_a = entity_list[i][entity_to_num[entity_a]]
                    entity_list_b = entity_list[i][entity_to_num[entity_b]]
                    all_pair = search_all_pair(entity_list_a,relation_name,entity_list_b)
                    # print (all_pair)
                    if(len(all_pair)!=0):
                        # 将三元组拼接成字符串
                        for p in all_pair:
                            str = "("+entity_a+": "+p[0]+","+relation_name+","+entity_b+": "+p[2]+")"
                            one_sentence_pair_list.append(str)
                        # print (all_pair)
            all_pair_list.append(one_sentence_pair_list)
        return all_pair_list

# 对于每个实体a, 关系 r, 只匹配最近的一个实体b
def search_all_pair(entity_list_a,relation,entity_list_b):
    tuple_list=[]
    for a in entity_list_a:
        entity_b =""
        closest_index=512
        for b in entity_list_b:
            if(a[0]!='nan' and b[0]!='nan' and a[0]!=b[0] and abs(b[1]-a[1]) < closest_index):
                closest_index = abs(b[1]-a[1])
                entity_b = b[0]
        if(closest_index!=512):
            one_tuple = [a[0], relation, entity_b]
            tuple_list.append(one_tuple)
    return tuple_list

# 输入文件是每一行十二个浮点数的文件，返回二维的关系list
def get_relation(relation_path):
    reader = open(relation_path, encoding='utf-8-sig')
    list_data = reader.readlines()
    relation_list=[]
    for i in list_data:
        i = i.replace('\n', '')
        all_rel_list=i.split(' ')
        all_rel_list = [i for i in all_rel_list if i != '']
        rel_list=[]
        for j in range(len(all_rel_list)):
            # print (j)
            if(float(all_rel_list[j])>0.5):
                rel_list.append(j)
        relation_list.append(rel_list)
    return relation_list

'''
    关系文件为每行12维的浮点数
'''
def main():
    relation_path="Fold1Result.txt"
    entity_path="PredictResult.xls"
    relation_list = get_relation(relation_path)
    entity_list=extract_entity(entity_path)
    tuple_list = generate_entity_relation(entity_list[0:1000],relation_list[0:1000])
    for t in range(len(tuple_list)):
        tuple_list[t] = ";".join(tuple_list[t])
    column = ['tuples']
    res = pd.DataFrame(columns=column, data=tuple_list)
    res.to_csv('tuple_result.csv',encoding='utf-8-sig')

if __name__ == '__main__':
    main()
    print ('now __name__ is %s' % __name__)