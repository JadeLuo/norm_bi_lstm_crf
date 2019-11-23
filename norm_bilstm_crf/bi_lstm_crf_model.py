#encoding=utf-8
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood
import bi_lstm_data_helper
class BiLSTM_CRF(object):
    def __init__(self,config):
        self.config=config
        #使用的为动态的RNN结构，句子的长度是动态获取的
        #第一维是batch_siz的大小，第二维是句子的长度都无法固定
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, None], name='input_y')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.drop_out = tf.placeholder(tf.float32, name="drop_out")
        self.bilstm_crf()
    def bilstm_crf(self):
        _,_,len_set_word,sen_len_list = bi_lstm_data_helper.process_file('data/train.txt','dict/zi.txt','dict/pseg.txt')
        with tf.name_scope("embedding"):
            print('begin_embedding')
            #随机初始化并设为可训练的
            init_word_vecs=np.random.uniform(-0.25,0.25,[len_set_word,self.config.embedding_size])
            init_word_vecs=tf.convert_to_tensor(init_word_vecs,dtype=tf.float32)
            word_vecs=tf.get_variable(name='init_embedding',dtype=tf.float32,
                                      initializer=init_word_vecs,trainable=True)
            self.input_x_content = tf.nn.embedding_lookup(word_vecs, self.input_x)
        with tf.name_scope("bi_lstm"):
            print('run_in_lstm-crf')
            cell_fw = LSTMCell(self.config.lstm_hidden_size)
            #cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.config.drop_out)
            cell_bw = LSTMCell(self.config.lstm_hidden_size)
            #cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.config.drop_out)
            # inputs(self.input_x)的shape通常是[batch_size, sequence_length, dim_embedding]
            # output_fw_seq和output_bw_seq的shape都是[batch_size, sequence_length, hidden_dim]
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.input_x_content,
                                                                                self.sequence_lengths, dtype=tf.float32)

            self.out_put = tf.concat([output_fw_seq, output_bw_seq], axis=-1)  # 对正反向的输出进行合并
            print(self.out_put)
            s_shape =tf.shape(self.out_put)
            self.out_put = tf.reshape(self.out_put, [-1, 2 * self.config.lstm_hidden_size])
            self.pred = tf.contrib.layers.fully_connected(self.out_put, self.config.num_tags, activation_fn=None)
            self.logits = tf.reshape(self.pred,[-1,s_shape[1],self.config.num_tags])
        with tf.name_scope("crf_loss"):
            print('run_in_cal_loss')
            self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.input_y,
                                                                   sequence_lengths=self.sequence_lengths)
            self.crf_loss=-tf.reduce_mean(self.log_likelihood)
        with tf.name_scope("optimize"):
            print('run_in_optimize')
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.config.init_learning_rate)
            grads_and_vars = optim.compute_gradients(self.crf_loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.config.clip_grad,self.config.clip_grad), v] for g, v in
                                   grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)