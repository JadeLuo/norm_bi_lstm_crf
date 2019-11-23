#encoding=utf-8
import pandas as pd
from collections import defaultdict
#获得训练格式的数据
df_label=pd.read_csv('data/Train_makeup_labels.csv')
df_data=pd.read_csv('data/Train_makeup_reviews.csv')
id_list=list(df_data['id'])
content_list=list(df_data['Reviews'])
id2reviews_dict=dict(zip(id_list,content_list))
label_id_list=list(df_label['id'])
AspectTerms_start_list=list(df_label['A_start'])
AspectTerms_end_list=list(df_label['A_end'])
OpinionTerms_start_list=list(df_label['O_start'])
OpinionTerms_end_list=list(df_label['O_end'])
Aspect_id2index=defaultdict(list)
Opinion_id2index=defaultdict(list)
for id,start,end in zip(label_id_list,AspectTerms_start_list,AspectTerms_end_list):
    if start!=' ':
        Aspect_id2index[id].append([int(start),int(end)])
for id,start,end in zip(label_id_list,OpinionTerms_start_list,OpinionTerms_end_list):
    if start!=' ':
        Opinion_id2index[id].append([int(start),int(end)])
tag_list=[]
for content in content_list:
    tag_list.append(['O']*len(content))
for key in Aspect_id2index:
    for index_list in Aspect_id2index[key]:
        start=index_list[0]
        end=index_list[1]
        for i in range(start,end):
            if i==start:
                tag_list[int(key)-1][i]='B-ASP'
            else:
                tag_list[int(key)-1][i] = 'I-ASP'
for key in Opinion_id2index:
    for index_list in Opinion_id2index[key]:
        start=index_list[0]
        end=index_list[1]
        for i in range(start,end):
            if i==start:
                tag_list[int(key)-1][i]='B-OPI'
            else:
                tag_list[int(key)-1][i] = 'I-OPI'
train_content=content_list[:-1300]
test_content=content_list[-1300:]
train_tag=tag_list[:-1300]
test_tag=tag_list[-1300:]
with open('data/train.txt','a+',encoding='utf-8') as fw:
    for i in range(len(train_content)):
        for char,tag in zip(train_content[i],train_tag[i]):
            data=char+'_s_'+tag
            fw.write(data+'\n')
        fw.write('\n')
with open('data/val.txt','a+',encoding='utf-8') as fw:
    for i in range(len(test_content)):
        for char,tag in zip(test_content[i],test_tag[i]):
            data=char+'_s_'+tag
            fw.write(data+'\n')
        fw.write('\n')
