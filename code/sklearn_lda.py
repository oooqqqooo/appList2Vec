# -*- coding: UTF-8 -*-
'''
frx
2018.3.8
'''
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
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

temp1,corpora_documents1 = read2doc('dvc_sample.txt')
temp2,corpora_documents2 = read2doc('dvc_sample1.txt')
#print('changdu---',len(corpora_documents),corpora_documents[5])

print('______zhe li')
#print(len(temp),temp.head())

def foo(model, feature_names, n_top_words, out):
    for topic_idx, topic in enumerate(model.components_):
        x = ('topic #%d' % topic_idx)
        y ='  '.join([feature_names[i] for i in topic.argsort()[: -n_top_words - 1: -1]])
        xy = x + '\n' +y +'\n'
        #print(type(xy))
        out = out + xy
        #print(out)
    return out

word1 = []
for i in corpora_documents1:
    word1.append(' '.join(i))
cntVect = CountVectorizer()
cntVect.fit(word1)
cntTf1 = cntVect.transform(word1)

word2 = []
for i in corpora_documents2:
    word2.append(' '.join(i))
cntTf2 = cntVect.transform(word2)

def writeResult(cntTf,temp,ss='total'):
    doc_topic_dist = lda.transform(cntTf)
    doc_topic_dist = pd.DataFrame(doc_topic_dist, columns=['topic_#%d' % i for i in range(topic_numb)])
    #doc_topic_dist = pd.join([doc_topic_dist,store_label])
    print(doc_topic_dist.head())
    doc_topic_dist = doc_topic_dist.join(temp['user'],how='right')
    print(doc_topic_dist.head())
    #print(store_label.head())
    wo = r'./temp/%s_lda_vec_n_%d.csv' % (ss,topic_numb)
    doc_topic_dist.to_csv(wo, index=False,encoding='utf-8')
    print('____完成一次循环',topic_numb)
    return None

#这里是用app_install_total.txt作为训练集训练lda模型，然后预测app_install_new.txt的主题分布
list_numb_topics = [300,500,1000]
for topic_numb in list_numb_topics:
    #topic_numb = 18
    lda = LatentDirichletAllocation(n_topics=topic_numb,max_iter=1000, learning_method='batch')
     #保存模型
    lda.fit(cntTf1)
    wo = r'lda_%d.pkl' % topic_numb
    f = open(wo, 'wb')
    pickle.dump(lda, f)
    f.close()

    dic = {}
    out = ""
    n_top_words = 20
    tf_features_names = cntVect.get_feature_names()
    out = foo(lda, tf_features_names, n_top_words, out)
    # print(out)
    wo = r'./temp/topic_num_%d.txt' % topic_numb
    file = open(wo, 'w')
    file.write(out)
    # print(dic)
    writeResult(cntTf1,temp1,ss='total')
    writeResult(cntTf1,temp1,ss='new')
