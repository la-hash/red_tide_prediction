import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn import svm
import random

csv_data = pd.read_csv('Index_All.csv')  # 注：代码文件和.csv放在同一目录下
Index=pd.DataFrame(csv_data)

#MCI部分
MCI = Index[['Class', 'MCI']]  # 选取其中的两列数据
MCI_algea = list()
MCI_clean = list()
for i in range(len(MCI)):
    if MCI['Class'][i] == 'algea':
        algea_MCI=MCI['MCI'][i]
        MCI_algea.append(algea_MCI)
    else:
        clean_MCI = MCI['MCI'][i]
        MCI_clean.append(clean_MCI)#计算MCI赤潮、非赤潮两分表
#list转化为array
array_MCI_algea = np.asarray(MCI_algea)
array_MCI_clean = np.asarray(MCI_clean)
# #绘图
# plt.scatter(range(len(MCI_algea)),array_MCI_algea,color='r')#红点赤潮
# plt.scatter(range(len(MCI_clean)),array_MCI_clean, color='b')#蓝点非赤潮
# plt.show()

#ThreeBand
ThreeBand = Index[['Class', 'ThreeBand']]  # 选取其中的两列数据
ThreeBand_algea = list()
ThreeBand_clean = list()
for i in range(len(ThreeBand)):
    if ThreeBand['Class'][i] == 'algea':
        algea_ThreeBand=ThreeBand['ThreeBand'][i]
        ThreeBand_algea.append(algea_ThreeBand)
    else:
        clean_ThreeBand = ThreeBand['ThreeBand'][i]
        ThreeBand_clean.append(clean_ThreeBand)#计算MCI赤潮、非赤潮两分表
#list转化为array
array_ThreeBand_algea = np.asarray(ThreeBand_algea)
array_ThreeBand_clean = np.asarray(ThreeBand_clean)

#Index1
Index1= Index[['Class', 'Index1']]  # 选取其中的两列数据
Index1_algea = list()
Index1_clean = list()
for i in range(len(Index1)):
    if Index1['Class'][i] == 'algea':
        algea_Index1=Index1['Index1'][i]
        Index1_algea.append(algea_Index1)
    else:
        clean_Index1 = Index1['Index1'][i]
        Index1_clean.append(clean_Index1)#计算MCI赤潮、非赤潮两分表
#list转化为array
array_Index1_algea = np.asarray(Index1_algea)
array_Index1_clean = np.asarray(Index1_clean)

#Index2
Index2 = Index[['Class', 'Index2']]  # 选取其中的两列数据
Index2_algea = list()
Index2_clean = list()
for i in range(len(Index2)):
    if Index2['Class'][i] == 'algea':
        algea_Index2=Index2['Index2'][i]
        Index2_algea.append(algea_Index2)
    else:
        clean_Index2 = Index2['Index2'][i]
        Index2_clean.append(clean_Index2)#计算MCI赤潮、非赤潮两分表
#list转化为array
array_Index2_algea = np.asarray(Index2_algea)
array_Index2_clean = np.asarray(Index2_clean)

#InfraredPeak
InfraredPeak = Index[['Class', 'InfraredPeak']]  # 选取其中的两列数据
InfraredPeak_algea = list()
InfraredPeak_clean = list()
for i in range(len(InfraredPeak)):
    if InfraredPeak['Class'][i] == 'algea':
        algea_InfraredPeak=InfraredPeak['InfraredPeak'][i]
        InfraredPeak_algea.append(algea_InfraredPeak)
    else:
        clean_InfraredPeak = InfraredPeak['InfraredPeak'][i]
        InfraredPeak_clean.append(clean_InfraredPeak)#计算MCI赤潮、非赤潮两分表
#list转化为array
array_InfraredPeak_algea = np.asarray(InfraredPeak_algea)
array_InfraredPeak_clean = np.asarray(InfraredPeak_clean)


#分位表部分



#分数表函数定义
arr = array_MCI_algea  #放入要求序列（第一分表）
def score_define(arr):
    score_index = []
    index_max=max(arr)
    index_min=min(arr)
    one_part=(index_max-index_min)/9
    score_index.append(index_min)
    score_index.append(index_min+one_part)
    score_index.append(index_min+2*one_part)
    score_index.append(index_min+3*one_part)
    score_index.append(index_min+4*one_part)
    score_index.append(index_min+5*one_part)
    score_index.append(index_min+6*one_part)
    score_index.append(index_min+7*one_part)
    score_index.append(index_min+8*one_part)
    score_index.append(index_max)
    return score_index

#分数表函数使用
score_MCI=score_define(array_MCI_algea)
print('score_MCI',score_MCI)
score_ThreeBand=score_define(array_ThreeBand_algea)
print('score_ThreeBand',score_ThreeBand)
score_Index1=score_define(array_Index1_algea)
print('score_Index1',score_Index1)
score_Index2=score_define(array_Index2_algea)
print('score_Index2',score_Index2)
score_InfraredPeak=score_define(array_InfraredPeak_algea)
print('score_InfraredPeak',score_InfraredPeak)






#计算每条数据得分
csv_file = "Index_All.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
Index_All = pd.DataFrame(csv_data)

scores_for_everyone_MCI = pd.DataFrame(columns=['Class','score1','score3','score4','score5'])
scores_for_everyone_ThreeBand= pd.DataFrame(columns=['Class','score2','score3','score4','score5'])
score_MCI_frame=pd.DataFrame(columns=['Class','score'])
score_ThreeBand_frame=pd.DataFrame(columns=['Class','score'])
#MCI得分
for i in range(len(Index_All)):
    Class=Index_All['Class'][i]
    if Index_All['MCI'][i]<score_MCI[0]:
        score1 = 10
    elif Index_All['MCI'][i] < score_MCI[1]:
        score1 = 9
    elif Index_All['MCI'][i] < score_MCI[2]:
        score1 = 8
    elif Index_All['MCI'][i] < score_MCI[3]:
        score1 = 7
    elif Index_All['MCI'][i] < score_MCI[4]:
        score1 = 6
    elif Index_All['MCI'][i] < score_MCI[5]:
        score1 = 5
    elif Index_All['MCI'][i] < score_MCI[6]:
        score1 = 4
    elif Index_All['MCI'][i] < score_MCI[7]:
        score1 = 3
    elif Index_All['MCI'][i] < score_MCI[8]:
        score1 = 2
    elif Index_All['MCI'][i] < score_MCI[9]:
        score1 = 1
    else:
        score1 = 0
#     scores_for_everyone_MCI=scores_for_everyone_MCI.append({'Class':Class,'score1':score1},ignore_index=True)
# #ThreeBand得分
# for i in range(len(Index_All)):
#     Class=Index_All['Class'][i]
    if Index_All['ThreeBand'][i]<score_ThreeBand[0]:
        score2 = 10
    elif Index_All['ThreeBand'][i] < score_ThreeBand[1]:
        score2 = 9
    elif Index_All['ThreeBand'][i] < score_ThreeBand[2]:
        score2 = 8
    elif Index_All['ThreeBand'][i] < score_ThreeBand[3]:
        score2 = 7
    elif Index_All['ThreeBand'][i] < score_ThreeBand[4]:
        score2 = 6
    elif Index_All['ThreeBand'][i] < score_ThreeBand[5]:
        score2 = 5
    elif Index_All['ThreeBand'][i] < score_ThreeBand[6]:
        score2 = 4
    elif Index_All['ThreeBand'][i] < score_ThreeBand[7]:
        score2 = 3
    elif Index_All['ThreeBand'][i] < score_ThreeBand[8]:
        score2 = 2
    elif Index_All['ThreeBand'][i] < score_ThreeBand[9]:
        score2 = 1
    else:
        score2 = 0
#     scores_for_everyone_ThreeBand=scores_for_everyone_ThreeBand.append({'Class':Class,'score2':score2},ignore_index=True)
# #Index1得分
# for i in range(len(Index_All)):
#     Class=Index_All['Class'][i]
    if Index_All['Index1'][i]<score_Index1[0]:
        score3 = 1
    elif Index_All['Index1'][i] < score_Index1[1]:
        score3 = 2
    elif Index_All['Index1'][i] < score_Index1[2]:
        score3 = 3
    elif Index_All['Index1'][i] < score_Index1[3]:
        score3 = 4
    elif Index_All['Index1'][i] < score_Index1[4]:
        score3 = 5
    elif Index_All['Index1'][i] < score_Index1[5]:
        score3 = 6
    elif Index_All['Index1'][i] < score_Index1[6]:
        score3 = 7
    elif Index_All['Index1'][i] < score_Index1[7]:
        score3 = 8
    elif Index_All['Index1'][i] < score_Index1[8]:
        score3 = 9
    else:
        score3 = 10
#     scores_for_everyone_MCI=scores_for_everyone_MCI.append({'Class': Class, 'score3': score3},ignore_index=True)
#     scores_for_everyone_ThreeBand=scores_for_everyone_ThreeBand.append({'Class': Class, 'score3': score3}, ignore_index=True)
# #Index2得分
# for i in range(len(Index_All)):
#     Class=Index_All['Class'][i]
    if Index_All['Index2'][i]<score_Index2[0]:
        score4 = 0
    elif Index_All['Index2'][i] < score_Index2[1]:
        score4 = 1
    elif Index_All['Index2'][i] < score_Index2[2]:
        score4 = 2
    elif Index_All['Index2'][i] < score_Index2[3]:
        score4 = 3
    elif Index_All['Index2'][i] < score_Index2[4]:
        score4 = 4
    elif Index_All['Index2'][i] < score_Index2[5]:
        score4 = 5
    elif Index_All['Index2'][i] < score_Index2[6]:
        score4 = 6
    elif Index_All['Index2'][i] < score_Index2[7]:
        score4 = 7
    elif Index_All['Index2'][i] < score_Index2[8]:
        score4 = 8
    elif Index_All['Index2'][i] < score_Index2[9]:
        score4 = 9
    else:
        score4 = 10
#     scores_for_everyone_MCI=scores_for_everyone_MCI.append({'Class': Class, 'score4': score4},ignore_index=True)
#     scores_for_everyone_ThreeBand=scores_for_everyone_ThreeBand.append({'Class': Class, 'score2': score4}, ignore_index=True)
#
# #InfraredPeak得分
# for i in range(len(Index_All)):
#     Class=Index_All['Class'][i]
    if Index_All['InfraredPeak'][i]<score_InfraredPeak[0]:
        score5 = 0
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[1]:
        score5 = 1
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[2]:
        score5 = 2
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[3]:
        score5 = 3
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[4]:
        score5 = 4
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[5]:
        score5 = 5
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[6]:
        score5 = 6
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[7]:
        score5 = 7
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[8]:
        score5 = 8
    elif Index_All['InfraredPeak'][i] < score_InfraredPeak[8]:
        score5 = 9
    else:
        score5 = 10
    scores_for_everyone_MCI=scores_for_everyone_MCI.append({'Class': Class,'score1': score1,'score3': score3,'score4': score4, 'score5': score5},ignore_index=True)
    scores_for_everyone_ThreeBand=scores_for_everyone_ThreeBand.append({'Class': Class, 'score2': score2,'score3': score3,'score4': score4, 'score5': score5}, ignore_index=True)
print(scores_for_everyone_MCI)
print(scores_for_everyone_ThreeBand)

# 写入文件
scores_for_everyone_MCI.to_csv("score_MCI.csv")
scores_for_everyone_ThreeBand.to_csv("score_ThreeBand.csv")

#支持向量机向量提取
x=[]
y=[]
x_test=[]
y_test=[]
x_train=[]
y_train=[]
for i in range(len(scores_for_everyone_MCI)):
    if  scores_for_everyone_MCI['Class'][i] == 'algea':
        y_single=1
    else:y_single=0
    #ThreeBand版
    #x_single = [scores_for_everyone_ThreeBand['score2'][i], scores_for_everyone_ThreeBand['score3'][i],
               # scores_for_everyone_ThreeBand['score4'][i]]
    #MCI版
    x_single=[scores_for_everyone_MCI['score1'][i],scores_for_everyone_MCI['score3'][i],scores_for_everyone_MCI['score4'][i]]

    x.append(x_single)
    y.append(y_single)
print(x)
print(y)
i_train=random.sample(range(len(x)),636)


for i in range(len(x)):
    if i in i_train:
        x_train.append(x[i])
        y_train.append(y[i])
    else:
        x_test.append(x[i])
        y_test.append(y[i])
print('x_train',x_train)
print('y_train',y_train)
print('x_test',x_test)
print('y_test',y_test)

# print(x_test)
# print(y_test)



#支持向量机具体运作部分
clf = svm.SVC(kernel='linear')  # SVM模块，svc,线性核函数
clf.fit(x_train, y_train)

print(clf)

print(clf.support_vectors_)  # 支持向量点

print(clf.support_)  # 支持向量点的索引

print(clf.n_support_)  # 每个class有几个支持向量点

print('prediction',clf.predict(x_test))

weight = clf.coef_[0]
bias = clf.intercept_[0]
print('----------------------------')
print('weight', weight)
print('bias',bias)
y_prediction=clf.predict(x_test)
    # print('prediction',(x_test[i]))
#test检验部分
right=0
wrong=0
for i in range(len(y_test)):
    if y_test[i]==y_prediction[i]:
        right+=1
    else:
        wrong+=1
Accuracy=right/(right+wrong)
print('Accuracy',Accuracy)
# #t检验部分
# score_MCI_frame_algea=[]
# score_MCI_frame_clean=[]
# for i in range(len(scores_for_everyone_MCI)):
#      score=w1_best*scores_for_everyone_MCI['score1'][i]+w3_best*scores_for_everyone_MCI['score3'][i]+w4_best*scores_for_everyone_MCI['score4'][i]+w5_best*scores_for_everyone_MCI['score5'][i]
#      print(score)
#      if scores_for_everyone_MCI['Class'][i]=='algea':
#         score_MCI_frame=score_MCI_frame.append(score)
#      else:
#          score_MCI_frame= score_MCI_frame.append( score)
#
# print(score_MCI_frame_algea)
# print(score_MCI_frame_clean)
# #
# # # # import scipy.stats
# # print(scipy.stats.levene(score_MCI_frame_algea['score'],score_MCI_frame_clean['score']))
# # # t, pval = scipy.stats.ttest_ind(score_MCI_frame_algea['score'],score_MCI_frame_clean['score'])
# # # print(t,pval)
#验证集（7:3）分？

