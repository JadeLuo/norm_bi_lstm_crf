#encoding=utf-8
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr
def get_data_and_label(file_loc):
    '''
    :param file_loc: txt文件的完整路径
    :return: 两个大的List，前面的list里面一个个小list存放的是一个个句子，后面的list里面一个个小list存放的是一个个句子对应的标签
    '''

    datas_list=[]
    labels_list=[]
    with open(file_loc,'r',encoding='utf-8') as fr:
        data_list=[]
        label_list = []
        for line in fr.readlines():
            new_line=line.strip()
            if new_line=='':
                datas_list.append(data_list)
                labels_list.append(label_list)
                data_list = []
                label_list = []
            else:
                data_list.append(new_line.split()[0])
                label_list.append(new_line.split()[1])
    return datas_list,labels_list
#根据训练语料构建词汇表以及词性表进行存储
def build_vocab_pseg_file(train_file_location,val_file_location,vocab_location,pseg_location,top_n=None,encoding='utf-8'):
    train_datas_list,_=get_data_and_label(train_file_location)
    val_datas_list,_=get_data_and_label(val_file_location)
    all_datas_list=[]
    all_datas_list.extend(train_datas_list)
    all_datas_list.extend(val_datas_list)
    all_words = []
    for content in all_datas_list:
        all_words.extend(content)
    #若所有的词都存入词汇表
    #额外添加'<PAD>'以及'UNKNOW'是为了padding 0以及padding不存在于词汇表中的词做准备
    if not top_n:
        all_words=list(set(all_words))
        words=['<PAD>']+all_words+['UNKNOW']
    #若只选取top_n放入词汇表
    else:
        counter = Counter(all_words)
        count_pairs = counter.most_common(min(len(all_words),top_n))
        temp_words,_ =zip(*count_pairs)
        temp_words=list(temp_words)
        words = ['<PAD>'] + temp_words + ['UNKNOW']
    with open(vocab_location,'w',encoding=encoding) as fw:
        for word in words:
            fw.write(word+'\n')
    pseg_list=['<PAD>'] + ['v','n','l','y','r','x'] + ['UNKNOW']
    with open(pseg_location,'w',encoding=encoding) as fw:
        for pseg in pseg_list:
            fw.write(pseg+'\n')
#获得训练语料的词表词性表以及词向id词性向id映射的字典
def get_word_and_pseg2id_dict(vocab_location,pseg_location,encoding='utf-8'):
    words=[]
    psegs=[]
    with open(vocab_location, 'r', encoding=encoding) as fr:
        for line in fr.readlines():
            new_line=line.strip('\n')
            words.append(new_line)
    with open(pseg_location, 'r', encoding=encoding) as fr:
        for line in fr.readlines():
            new_line=line.strip('\n')
            psegs.append(new_line)
    word2id_dict = dict(zip(words,range(len(words))))
    pseg2id_dict = dict(zip(psegs,range(len(psegs))))
    return words,word2id_dict,psegs,pseg2id_dict
#用于字标注向索引的转换，因为tensorflow中的crf接收的数据是数字形式的索引
def tags2id(tags_list,tag2label):
    '''
    :param tags_list:
    :param tag2label:字标注向索引映射的字典
    :return:
    '''
    tags_label_list=[]
    for tags in tags_list:
        tags_label=[]
        for tag in tags:
            tags_label.append(tag2label[tag])
        tags_label_list.append(tags_label)
    print('final tags2id')
    return tags_label_list
#对训练数据进行处理，返回词表中各词对应的词向量,padding后语料向id的转换以及one_hot的label
def process_file(file_location,vocab_dic_location,psegs_dic_location):
    datas_list,labels_list=get_data_and_label(file_location)
    tag2label_dict = {"O": 0,
                 "B-PRO": 1, "I-PRO": 2,
                 "B-ATT": 3, "I-ATT": 4
                 }
    # 按文本中最长的句子长度进行填充
    sen_max_len = max(map(lambda x: len(x),datas_list))
    words, word2id_dict, psegs, pseg2id_dict = get_word_and_pseg2id_dict(vocab_dic_location,psegs_dic_location)
    len_set_word=len(list(word2id_dict.keys()))
    #把文本中词向id转换
    sen_index = []
    tag_index=[]
    sen_len_list=[]
    for sen in datas_list:
        words_index = []
        sen_len_list.append(min(len(sen), sen_max_len))
        for word in sen:
            if word in word2id_dict:
                words_index.append(word2id_dict[word])
            #若只选取top_n加入词表，则会存在词表中不存在的词都把其id设置为UNKNOW
            else:
                words_index.append(word2id_dict['UNKNOW'])
        sen_index.append(words_index)
    # 这里进行的是padding操作，根据设定的句子的最大长度，长的截取，短的补0
    # 注意：1.sen_index中把词转换为相应的id，2.默认的为从前面填充0,3.默认的长度过长的从前面截取
    #注意这里一定要改，默认从前面填充，但是动态的RNN是从后面截取长度
    sen_index = kr.preprocessing.sequence.pad_sequences(sen_index,sen_max_len,padding='post',truncating='post')

    for tags in labels_list:
        tags_index = []
        for tag in tags:
            tags_index.append(tag2label_dict[tag])
        tag_index.append(tags_index)
    # 这里进行的是padding操作，根据设定的句子的最大长度，长的截取，短的补0
    # 注意：1.sen_index中把词转换为相应的id，2.默认的为从前面填充0,3.默认的长度过长的从前面截取
    tag_index = kr.preprocessing.sequence.pad_sequences(tag_index,sen_max_len,padding='post',truncating='post')
    return sen_index,tag_index,len_set_word,np.array(sen_len_list)
#生成批次数据
def batch_iter(x,y,seq_len,batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    #打乱数据，若数据已打乱则不需要
    # indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    #每次返回一个批次的词、词性以及类别标签
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size,data_len)
        yield x[start_id:end_id],y[start_id:end_id],seq_len[start_id:end_id]
if __name__ == '__main__':
    build_vocab_pseg_file('data/train.txt','data/val.txt','dict/zi.txt','dict/pseg.txt')
    # words, word2id_dict, psegs, pseg2id_dict=get_word_and_pseg2id_dict('dict/zi.txt','dict/pseg.txt')
    # print(word2id_dict)
    # print(pseg2id_dict)