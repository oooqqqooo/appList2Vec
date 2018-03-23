# -*- coding: UTF-8 -*-
'''
frx
2018.3.8
因为用户不唯一，存在相同用户操作时间不同，所以有不同列表，这里对于lsiToVec.py文件作出改进，合并相同用户
'''
from gensim import corpora,models
import pandas as pd
import numpy as np


df = pd.read_csv('./out/app_top59.csv',header=None,encoding='utf-8')
app_top100 = np.array(df)

f1 = open('./out/app_install_total.txt',encoding='utf-8')
f2 = open('./out/app_install_new.txt',encoding='utf-8')

line = []
df = []
list_app_user = [];list_app = [];x = []
corpora_documents1 = []
xxx = 0
for i in f1.readlines():
    # print(i)
    line = i.strip().split('\t')
    #读到的数据为str类型，需要用eval()将str转化为列表数据
    line[3] = eval(line[3])
    for j, dic in enumerate(list(line[3])):
        if dic['app_name'] in app_top100:
            xxx = xxx + 1
        else:
            for k in range(len(list(dic['load_info']))):
                list_app.append(dic['app_name'])
            # print('list_app',list_app)
            for l in list_app:
                list_app_user.append(l)
            # print('list_app_user',list_app_user)
            list_app[:] = []
    #合并相同用户
    if line[1] in x:
        for n in list_app_user:
            corpora_documents1[x.index(line[1])].append(n)
        list_app_user[:] = []
    else:

        #这里corpora_documents.append(list(list_app_user))必须加list（），不加的话清空list_app_user内存连corpora_documents保存的内容也会被清空
        corpora_documents1.append(list(list_app_user))
        #print('corpora_documents',corpora_documents)
        list_app_user[:] = []
        x.append(line[1])
        df.append(line)
f1.close()
temp1 = pd.DataFrame(df,columns=['dvc','user','time','list'])


line = []
df2 = []
list_app_user2 = [];list_app2 = [];x = []
corpora_documents2 = []
xx = 0
for i in f2.readlines():
    #print(i)
    line = i.strip().split('\t')
    #读到的数据为str类型，需要用eval()将str转化为列表数据
    line[3] = eval(line[3])
    for j, dic in enumerate(list(line[3])):
        if dic['app_name'] in app_top100:
            xx = xx + 1
        else:
            for k in range(len(list(dic['load_info']))):
                list_app2.append(dic['app_name'])
            # print('list_app',list_app)
            for l in list_app2:
                list_app_user2.append(l)
            # print('list_app_user',list_app_user)
            list_app2[:] = []
    #合并相同用户
    if line[1] in x:
        for n in list_app_user2:
            corpora_documents2[x.index(line[1])].append(n)
        list_app_user2[:] = []
    else:

        #这里corpora_documents.append(list(list_app_user))必须加list（），不加的话清空list_app_user内存连corpora_documents保存的内容也会被清空
        corpora_documents2.append(list(list_app_user2))
        #print('corpora_documents',corpora_documents)
        list_app_user2[:] = []
        x.append(line[1])
        df2.append(line)
f2.close()
#print('changdu---',len(corpora_documents),corpora_documents[5])
temp2 = pd.DataFrame(df2,columns=['dvc','user','time','list'])


# 生成字典和向量语料
dictionary1 = corpora.Dictionary(corpora_documents1)
print("字典：",dictionary1)
dictionary1.save('dict1.txt') #保存生成的词典


# 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
corpus1 = [dictionary1.doc2bow(text) for text in corpora_documents1]
corpus2 = [dictionary1.doc2bow(text) for text in corpora_documents2]
# 向量的每一个元素代表了一个word在这篇文档中出现的次数

# corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
tfidf_model = models.TfidfModel(corpus1)
corpus_tfidf1 = tfidf_model[corpus1]
# tfidf_model = models.TfidfModel(corpus2)
corpus_tfidf2 = tfidf_model[corpus2]
#print('@@@',list(corpus_tfidf))
print('开始跑模型----')
l = [300,500,1000]
for d in l:
    lsi_model = models.LsiModel(corpus_tfidf1, id2word=dictionary1, num_topics=d)
    wo = r'lsi_%d.model' % d
    lsi_model.save(wo)
    #lsi_model = models.LsiModel.load('lsi_1.model')
    #tmp_fname = get_tmpfile("lsi.model")
    #lsi_model = models.LsiModel.load(tmp_fname)
    corpus_lsi1 = lsi_model[corpus_tfidf1]
    corpus_lsi2 = lsi_model[corpus_tfidf2]
    #print(list(corpus_lsi))
    print('------')
    store1 = [];store2 = []
    

    for doc in corpus_lsi1:
        # 有的主题取值为0，默认不显示，所以这里定义定长列表接收主题向量值，没有的为0
        l = []
        for i in range(d):
            l.append(0)
        # print("doc:", doc)
        # if index == 2987:
        for ii in doc:
            l[ii[0]] = ii[1]
        store1.append(l)
    

    df = pd.DataFrame(store1, columns=['topic @%d' % ii for ii in range(d)])

    df = df.join(temp1['user'], how='right')
    print(df.head())
    print(df.shape)
    wo = r'total_num_topics_%d.csv' % d
    df.to_csv(wo, index=False, encoding='utf-8')

    for doc in corpus_lsi2:
        # 有的主题取值为0，默认不显示，所以这里定义定长列表接收主题向量值，没有的为0
        l = []
        for i in range(d):
            l.append(0)
        # print("doc:", doc)
        # if index == 2987:
        for ii in doc:
            l[ii[0]] = ii[1]
        store2.append(l)
    
    df = pd.DataFrame(store2, columns=['topic @%d' % ii for ii in range(d)])

    df = df.join(temp2['user'], how='right')
    print(df.head())
    print(df.shape)
    wo = r'new_num_topics_%d.csv' % d
    df.to_csv(wo, index=False, encoding='utf-8')
    print('一轮跑完-----')
