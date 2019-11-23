#encoding=utf-8
#encoding=utf-8
import tensorflow as tf
import time
import os
import numpy as np
from datetime import timedelta
from sklearn.metrics import classification_report,confusion_matrix
import bi_lstm_data_helper
from bi_lstm_crf_model import BiLSTM_CRF
from bi_lstm_crf_config import Config
#网络模型的保存文件夹
model_save_location="checkpoints/bilstm_crf"
#网络模型保存的相对路径
save_path = os.path.join(model_save_location, 'model.ckpt')
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif
    # return timedelta(seconds=int(round(time_dif)))
def train_tes_model(model,bilstm_crf_config):
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/bilstm_crf'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    #删除原来已存在的tensorboard文件
    else:
        file_list=os.listdir(tensorboard_dir)
        if len(file_list)>0:
            for file in file_list:
                os.remove(os.path.join(tensorboard_dir,file))
    tf.summary.scalar("loss", model.crf_loss)
    # tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver，用以保存模型
    saver = tf.train.Saver()
    if not os.path.exists(model_save_location):
        os.makedirs(model_save_location)
    #获得训练数据和测试数据
    start_time = time.time()
    sen_index, tag_index, len_set_word,sen_len_list= bi_lstm_data_helper.process_file('data/train.txt','dict/zi.txt','dict/pseg.txt')
    time_dif = get_time_dif(start_time)
    print("load data usage:",time_dif)
    print('Training and Testing...')
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer.add_graph(sess.graph)
        # 若存在检测点则在检测点的基础上继续训练
        # 获得所有检测点名称
        ckpt = tf.train.get_checkpoint_state(model_save_location)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        #批量获得数据
        global_step=0
        for epoch in range(bilstm_crf_config.num_epochs):
            batch_train = bi_lstm_data_helper.batch_iter(sen_index,tag_index,sen_len_list,bilstm_crf_config.batch_size)
            total_batch = 0
            for x_batch,y_batch,seq_len_batch in batch_train:
                total_batch+=1
                global_step += 1
                feed_dict={model.input_x:x_batch,model.input_y:y_batch,model.sequence_lengths:seq_len_batch,model.drop_out:bilstm_crf_config.drop_out}
                if global_step%bilstm_crf_config.save_per_batch==0:
                    saver.save(sess,save_path,global_step=global_step)
                    summary_str = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(summary_str,total_batch)  # 将summary 写入文件
                if total_batch%bilstm_crf_config.print_per_batch == 0:
                    loss = model.crf_loss.eval(feed_dict=feed_dict)
                    now_time_dif=get_time_dif(start_time)
                    print("Epoch %d:Step %d loss is %f" % (epoch+1,total_batch,loss))
                    print('Cost_time：{}'.format(now_time_dif))
                    start_time=time.time()
                sess.run(model.train_op, feed_dict=feed_dict)
        saver.save(sess,save_path,global_step=global_step)
        #训练完之后通过测试集测试模型
        from tensorflow.contrib.crf import viterbi_decode
        test_sen_index, test_tag_index,_, test_sen_len_list = bi_lstm_data_helper.process_file('data/val.txt',
                                                                                            'dict/zi.txt', 'dict/pseg.txt')
        batch_test =bi_lstm_data_helper.batch_iter(test_sen_index,test_tag_index,test_sen_len_list,bilstm_crf_config.batch_size)
        label_list = []
        for test_x_batch,test_y_batch,test_seq_len_batch in batch_test:
            feed_dict = {model.input_x: test_x_batch, model.input_y: test_y_batch, model.sequence_lengths: test_seq_len_batch,
                         model.drop_out:1.0}
            logits, transition_params = sess.run([model.logits,model.transition_params],feed_dict=feed_dict)

            for logit,seq_len in zip(logits, test_seq_len_batch):
                # viterbi_decode通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
                # viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表.
                # viterbi_score: 序列对应的概率值
                # 这是解码的过程，利用维特比算法结合转移概率矩阵和节点以及边上的特征函数求得最大概率的标注
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
        # 索引向标签的转换
        tag2label_dict = {"O": 0,
                          "B-PRO": 1, "I-PRO": 2,
                          "B-ATT": 3, "I-ATT": 4
                          }
        label2tag = {}
        for label, tag in tag2label_dict.items():
            label2tag[tag] = label
        tags_list = []
        final_test_tags_list=[]
        for labels in label_list:
            tags = []
            for i in labels:
                tags.append(label2tag[i])
            tags_list.append(tags)
        for labels in test_tag_index:
            tags = []
            for i in labels:
                tags.append(label2tag[i])
            final_test_tags_list.append(tags)
        # 产生标注报告,最好的结果是0.78不如CRF的0.82
        true_list = []
        pre_list = []
        for pre_tags, test_tags in zip(tags_list,final_test_tags_list):
            for pre_tag, test_tag in zip(pre_tags, test_tags):
                if pre_tag == test_tag == 'O':
                    pass
                else:
                    true_list.append(test_tag)
                    pre_list.append(pre_tag)
        print(classification_report(true_list, pre_list))
if __name__ == '__main__':
    bilstm_crf_config=Config()
    model=BiLSTM_CRF(bilstm_crf_config)
    train_tes_model(model,bilstm_crf_config)