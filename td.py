# coding: utf-8
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('agg')

import matplotlib.pyplot as plt

import tensorflow as tf
import time
from tensorflow.contrib.layers import dropout
from collections import namedtuple

# import sys
# import importlib
# importlib.reload(sys)



HParams = namedtuple('HParams',
                     'seq_size, hidden_size, batch_size, learning_rate, input_size, output_size')

# 通用权值初始化
def weight_init(shape):
    stddev = 2 / np.sqrt(shape[0])
    # Xavier初始化,截断正态分布
    weight_init = tf.truncated_normal(shape=shape, stddev=stddev)
    return tf.Variable(weight_init)


# 通用偏置初始化
def bias_init(shape):
    bias_init = tf.constant(0.1, shape=shape)
    return tf.Variable(bias_init)


# Dense层
def DNN(input_data, weight, bias, activefunction=None):
    output = tf.add(tf.matmul(input_data, weight), bias)
    if activefunction == None:
        return output
    else:
        return activefunction(output)


# LSTM网络
class RNN_LSTM(object):
    def __init__(self, hps, keep_prob=1, lstm_keep_prob=[1, 1, 1]):
        self._X = X = tf.placeholder(tf.float32, shape=[None, hps.seq_size, hps.input_size], name='X')
        self._y = y = tf.placeholder(tf.float32, shape=[None, hps.seq_size, hps.output_size], name='y')
        self._is_training = is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.keep_prob = keep_prob
        self.lstm_keep_prob = lstm_keep_prob

        # 输入层drop
        with tf.name_scope('Dropout'):
            X_drop = dropout(X, keep_prob=keep_prob, is_training=is_training)

        # lstm_cell，hidden层drop
        with tf.name_scope('LSTM_hidden'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hps.hidden_size, activation=tf.nn.elu)
            if is_training == True:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=lstm_keep_prob[0],
                                                          output_keep_prob=lstm_keep_prob[1],
                                                          state_keep_prob=lstm_keep_prob[2])
            else:
                pass
            rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_drop, dtype=tf.float32)

        # Dense网络,输出层
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, hps.hidden_size])
        with tf.name_scope('Dense'):
            W = weight_init(shape=[hps.hidden_size, hps.output_size])
            b = bias_init(shape=[hps.output_size])
            stacked_outputs = DNN(input_data=stacked_rnn_outputs, weight=W, bias=b)
            self._output = output = tf.reshape(stacked_outputs, shape=([-1, hps.seq_size, hps.output_size]))

        # 损失函数
        with tf.name_scope('loss'):
            self._cost = cost = tf.reduce_mean(tf.square(output - y))

        # 训练优化
        with tf.name_scope('train'):
            self._train_op = tf.train.AdamOptimizer(hps.learning_rate).minimize(cost)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._y

    @property
    def is_training(self):
        return self._is_training

    @property
    def cost(self):
        return self._cost

    @property
    def output(self):
        return self._output

    @property
    def train_op(self):
        return self._train_op

# 数据及训练过程
def train_test(hps, data):
    # 训练集数据
    train_data_len = len(data) * 2 // 3
    train_X, train_y = [], []
    for i in range(train_data_len - hps.seq_size - 1):
        X = data[i: i + hps.seq_size].tolist()
        y = data[i + 1: i + hps.seq_size + 1].tolist()
        train_X.append(X)
        train_y.append(y)
    train_X = np.reshape(train_X, newshape=([-1, hps.seq_size, hps.input_size]))
    train_y = np.reshape(train_y, newshape=([-1, hps.seq_size, hps.output_size]))

    # 测试集数据
    test_data_len = len(data) // 3
    test_X, test_y = [], []
    for i in range(train_data_len, train_data_len + test_data_len - hps.seq_size - 1):
        X = data[i: i + hps.seq_size].tolist()
        y = data[i + 1: i + hps.seq_size + 1].tolist()
        test_X.append(X)
        test_y.append(y)
    test_X = np.reshape(test_X, newshape=([-1, hps.seq_size, hps.input_size]))
    test_y = np.reshape(test_y, newshape=([-1, hps.seq_size, hps.output_size]))
    true_power = test_y[-1]

    # 训练过程
    with tf.Graph().as_default(), tf.Session() as sess:
        # with tf.device('/gpu:0'):
            # 初始化LSTM
            with tf.variable_scope('model', reuse=None):
                m_train = RNN_LSTM(hps, keep_prob=0.8, lstm_keep_prob=[0.5, 0.5, 0.5])

            init = tf.global_variables_initializer()
            sess.run(init)

            batch = len(train_X) // hps.batch_size
            # 迭代轮数
            for step in range(20000):
                # 进行批量训练
                for iteration in range(batch):
                    batch_start = iteration*hps.batch_size
                    batch_end = batch_start + hps.batch_size - 1
                    sess.run(m_train.train_op,
                            feed_dict={m_train.X: train_X[batch_start: batch_end],
                                        m_train.Y: train_y[batch_start: batch_end],
                                        m_train.is_training: True})
                train_cost = sess.run(m_train.cost,
                                    feed_dict={m_train.X: train_X, m_train.Y: train_y, m_train.is_training:False})
            test_output, test_cost = sess.run([m_train.output, m_train.cost],
                                            feed_dict={m_train.X: test_X, m_train.Y: test_y,
                                                       m_train.is_training: False})
            return train_cost, test_cost, test_output, true_power


def main():
    data = pd.read_excel('F:/毕设/数据/新/6030/6030.xlsx')
    data_19 = data[['Time', 'Power(dBm)']]
    # 转化为一维数据
    data_19 = np.array(data_19['Power(dBm)'])
    # 数据正则化
    normalized_data = (data_19 - np.mean(data_19)) / np.std(data_19)

    # 参数设置
    batch_size = 20
    learning_rate = 0.001
    input_size = 1
    output_size = 1
    costs = []
    times = []
    count = 0
    for seq_size in range(3, 31, 3):
        for hidden_size in [1, 2, 5, 7]:
            print(seq_size, hidden_size)
            hps = HParams(seq_size=seq_size, hidden_size=hidden_size, batch_size=batch_size,
                          learning_rate=learning_rate, input_size=input_size, output_size=output_size)
            t_start = time.time()
            train_cost, test_cost, pred, true_power = train_test(hps, normalized_data)
            t_end = time.time()
            print('Done')
            costs.append([train_cost, test_cost])
            times.append(t_end - t_start)

            # 数据绘制
            pred_power, true_power = np.array(pred[-1]), np.array(true_power)
            t = range(len(pred_power))
            plt.figure(count)
            plt.plot(t, pred_power[:, 0], '--', label='Prediction Power', marker='+')
            plt.plot(t, true_power[:, 0], label='True Power', marker='+')
            plt.title('Prediction')
            plt.xlabel('Time')
            plt.ylabel('Power')
            plt.legend()
            plt.savefig('F:/毕设/数据/新/6030/Prediction' + str(count) + '.png')
            plt.close()

            count+=1

    # 数据绘制
    plt.figure(count)
    plt.plot(data_19, 'r--', marker='d', label='original data')
    plt.plot(normalized_data, 'b--', marker='+', label='normalized data')
    plt.xlabel('Time')
    plt.ylabel('Power(dBm)')
    plt.title('Normalized data process')
    plt.legend()
    plt.savefig('F:/毕设/数据/新/6030/Normalized_data.png')
    plt.close()

    # 绘制误差结果图
    costs = np.array(costs)
    t = range(len(costs))
    plt.figure(count+1)
    plt.plot(t, costs[:, 0], label='train_cost', marker='+')
    plt.plot(t, costs[:, 1], '--', label='test_cost', marker='+')
    plt.title('Train and test error')
    plt.xlabel('Model')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig('F:/毕设/数据/新/6030/Error.png')
    plt.close()
    print(costs)

    # 绘制训练时间图
    times = np.array(times)
    t = range(len(times))
    plt.figure(count+2)
    plt.plot(t, times, label='Time', marker='+')
    plt.title('Model training time')
    plt.xlabel('Model')
    plt.ylabel('Using Time')
    plt.legend()
    plt.savefig('F:/毕设/数据/新/6030/Time.png')
    plt.close()

if __name__ == '__main__':
    main()
    print('Program is finished!')