# -*- coding: UTF-8 -*-
'''
frx
2018.3.8
因为用户不唯一，存在相同用户操作时间不同，所以有不同列表，这里对于lsiToVec.py文件作出改进，合并相同用户
'''
from gensim import corpora,models
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle

df = pd.read_csv('app_top59.csv',header=None,encoding='utf-8')
app_top100 = np.array(df)

def read2doc(path):
    f = open(path,encoding='utf-8')
    line = [];df = [];list_app_user = [];list_app = [];x = []
    corpora_documents = []
    xxx = 0
    for i in f.readlines():
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
                corpora_documents[x.index(line[1])].append(n)
            list_app_user[:] = []
        else:
            #这里corpora_documents.append(list(list_app_user))必须加list（），不加的话清空list_app_user内存连corpora_documents保存的内容也会被清空
            corpora_documents.append(list(list_app_user))
            #print('corpora_documents',corpora_documents)
            list_app_user[:] = []
            x.append(line[1])
            df.append(line)
    f.close()
    temp = pd.DataFrame(df, columns=['dvc', 'user', 'time', 'list'])
    return temp,corpora_documents

temp1,corpora_documents1 = read2doc('app_install_total.txt')
temp2,corpora_documents2 = read2doc('app_install_new.txt')
# 生成字典和向量语料
dictionary1 = corpora.Dictionary(corpora_documents1)
print("字典：",dictionary1)
dictionary1.save('.\install_lda\dict1.txt') #保存生成的词典

# 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
corpus1 = [dictionary1.doc2bow(text) for text in corpora_documents1]
corpus2 = [dictionary1.doc2bow(text) for text in corpora_documents2]
# 向量的每一个元素代表了一个word在这篇文档中出现的次数

# corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
tfidf_model = models.TfidfModel(corpus1)
corpus_tfidf1 = tfidf_model[corpus1]
#tfidf_model = models.TfidfModel(corpus2)
corpus_tfidf2 = tfidf_model[corpus2]


def foo(model, out):
    for i in model.print_topics(num_topics=1000, num_words=20):
        topic_idx = list(i)[0]
        x = ('topic #%d' % topic_idx)
        y =''.join(list(i)[1])
        xy = x + '\n' +y +'\n'
        #print(type(xy))
        out = out + xy
    return out

l = [1000];mm = []
for d in l:
    lda_model = models.LdaModel(corpus1,id2word=dictionary1,num_topics=d)
    wo = r'.\install_lda\lda_%d.model' % d
    lda_model.save(wo)
#文档在某些主题上概率较小，默认保存为0，这里设置参数minimum_probability=0.0000000001可以将小概率的值也保存下来，解决了输出文档-主题概率矩阵大多是0的问题
    corpus_lda1 = lda_model.get_document_topics(corpus1,minimum_probability=0.0000000001)
    corpus_lda2 = lda_model.get_document_topics(corpus2, minimum_probability=0.0000000001)

    print('------')
    store1 = [];store2 = []
    for doc in corpus_lda1:
        #有的主题取值为0，默认不显示，所以这里定义定长列表接收主题向量值，没有的为0
        l = []
        for i in range(d):
            l.append(0)
        for ii in doc:
            l[ii[0]] = ii[1]
        store1.append(l)

    df = pd.DataFrame(store1, columns=['topic @%d' % ii for ii in range(d)])
    df = df.join(temp1['user'], how='right')
    print(df.head())
    print(df.shape)
    wo = r'.\install_lda\total_num_topic= %d.csv' % d
    df.to_csv(wo, index=False, encoding='utf-8')

    for doc in corpus_lda2:
        # 有的主题取值为0，默认不显示，所以这里定义定长列表接收主题向量值，没有的为0
        l = []
        for i in range(d):
            l.append(0)
        for ii in doc:
            l[ii[0]] = ii[1]
        store2.append(l)

    df2 = pd.DataFrame(store2, columns=['topic @%d' % ii for ii in range(d)])
    df2 = df2.join(temp2['user'], how='right')
    print(df2.head())
    print(df2.shape)
    wo = r'.\install_lda\new_num_topic= %d.csv' % d
    df2.to_csv(wo, index=False, encoding='utf-8')

    out = ""
    out = foo(lda_model,out)
    wo = r'.\install_lda\topic_num%d.txt' % d
    file = open(wo, 'w')
    #出现编码问题，已解决，out .encode('GBK','ignore').decode('GBk')
    file.write(out .encode('GBK','ignore').decode('GBk'))

