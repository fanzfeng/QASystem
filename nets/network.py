# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng
from keras.layers import Dot

import os, sys
root_path = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
print(root_path)
sys.path.append(root_path)
from nets.utils_net import *


class TextCnn(object):
    def __init__(self, conf):
        isinstance(conf, dict)
        self.vocab_size = conf.get("vocab_size")
        self.input_len = conf["input_len"]
        self.emb_size = conf.get("emb_size")
        self.n_class = conf.get("num_class", 2)
        self.kernel_num = conf.get("kernel_num", 128)
        self.win_size_list = conf.get("win_size_list", [1, 2, 3, 4, 5])
        self.l2_reg_lambda = conf.get("l2_reg_lambda", 0.01)

        if self.vocab_size is not None:
            self.emb_layer = EmbeddingLayer(self.vocab_size, self.emb_size)
        self.num_filters_total = len(self.win_size_list)*self.kernel_num
        self.W_loss = tf.get_variable("W_loss", shape=[self.num_filters_total, self.n_class],
                                      initializer=tf.contrib.layers.xavier_initializer())
        self.b_loss = tf.Variable(tf.constant(0.1, shape=[self.n_class]), name="b_loss")

    def net(self, tensor_dict):
        input_tf = tensor_dict["input"]
        output_tf = tensor_dict["output"]
        keep_prob = tensor_dict["keep_rate"]
        if isinstance(input_tf, list):
            input_tf = input_tf[0]
        print("warning: only first vector is valid !!")
        if self.vocab_size is not None:
            input_tf = self.emb_layer.ops(input_tf)
        input_net = tf.expand_dims(input_tf, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.win_size_list):
            filter_shape = [filter_size, self.emb_size, 1, self.kernel_num]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.kernel_num]), name="b")
            conv = tf.nn.conv2d(input_net, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, self.input_len-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name="pool")
            pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, 3)
        out_pool = tf.reshape(h_pool, [-1, self.num_filters_total])
        h_drop = tf.nn.dropout(out_pool, keep_prob)

        l2_loss = tf.constant(0.0) + tf.nn.l2_loss(self.W_loss) + tf.nn.l2_loss(self.b_loss)
        scores = tf.nn.xw_plus_b(h_drop, self.W_loss, self.b_loss, name="scores")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=output_tf)) + \
               self.l2_reg_lambda * l2_loss
        return tf.nn.softmax(scores), output_tf, loss


class BertClassifier(object):
    def __init__(self, conf):
        assert isinstance(conf, dict)
        self.hidden_size = conf["hidden_size"]
        self.initializer_range = conf.get("initializer_range", 0.02)
        self.num_class = conf.get("num_class", 2)
        self.output_weights = tf.get_variable("output_weights", shape=[self.num_class, self.hidden_size],
                                              initializer=create_initializer(self.initializer_range))
        self.output_bias = tf.get_variable("output_bias", shape=[self.num_class], initializer=tf.zeros_initializer())
        self.pooled_output = None

    def net(self, tensor_dict):
        x_input = tensor_dict["input"]
        labels = tensor_dict["output"]
        keep_rate = tensor_dict["keep_rate"]
        if isinstance(x_input, list):
            x_input = x_input[0]
        if len(x_input.get_shape()) == 3:
            x_input = tf.squeeze(x_input[:, 0:1, :], axis=1)
        print("warning: only first vector is valid !!")
        self.pooled_output = tf.layers.dense(x_input, self.hidden_size, activation=tf.tanh,
                                             kernel_initializer=create_initializer(self.initializer_range))
        logits = tf.matmul(self.pooled_output, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        probabilities = tf.nn.dropout(logits, keep_prob=keep_rate)
        # if self.num_class == 2:
        #     one_hot_labels = tf.one_hot(labels, depth=self.num_class, dtype=tf.float32)
        #     per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=probabilities)
        #     loss = tf.reduce_mean(per_example_loss)
        # else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=probabilities))
        return tf.nn.softmax(probabilities), labels, loss


class DSSM(object):
    def __init__(self, conf):
        self.vocab_size = conf["vocab_size"]
        self.emb_size = conf["emb_size"]
        self.hidden_sizes = conf.get("hidden_sizes", [512, 768])
        self.n_class = conf["n_class"]
        if self.vocab_size and self.emb_size:
            self.emb_layer = EmbeddingLayer(self.vocab_size, self.emb_size)
        self.seq_pool_layer = SequencePoolingLayer()
        self.softsign_layer = SoftsignLayer()
        self.bow_layers = []
        for i in range(len(self.hidden_sizes)):
            if i < 1:
                self.bow_layers.append(FCLayer(self.emb_size, self.hidden_sizes[i]))
            else:
                self.bow_layers.append(FCLayer(self.hidden_sizes[i-1], self.hidden_sizes[i]))
        self.fc_layer = FCLayer(1, self.n_class)

    def mlp(self, x):
        y = x
        for u in self.bow_layers:
            y = u.ops(y)
            y = tf.nn.tanh(y)
        return y

    def net(self, tensor_dict):
        x_input = tensor_dict["input"]
        labels = tensor_dict["output"]
        keep_rate = tensor_dict["keep_rate"]
        assert isinstance(x_input, list)
        left, right = x_input[0:2]
        print("warning: only first vector is valid !!")
        if len(left.get_shape()) == 2:
            left_emb = self.emb_layer.ops(left)
            right_emb = self.emb_layer.ops(right)
        elif len(left.get_shape()) == 3:
            left_emb, right_emb = left, right
        left_pool = self.seq_pool_layer.ops(left_emb)
        right_pool = self.seq_pool_layer.ops(right_emb)
        left_soft = self.softsign_layer.ops(left_pool)
        right_soft = self.softsign_layer.ops(right_pool)
        left_mlp = self.mlp(left_soft)
        right_mlp = self.mlp(right_soft)
        print("before cross: ", left_mlp.get_shape())
        # cross = tf.tensordot(left_mlp, right_mlp, axes=[1, 1])
        cross = Dot(axes=[1,1], normalize=True)([left_mlp, right_mlp])
        print("after cross: ", cross.get_shape())
        logits = self.fc_layer.ops(cross)
        probabilities = tf.nn.dropout(logits, keep_prob=keep_rate)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probabilities, labels=labels))
        return tf.nn.softmax(probabilities), labels, loss


class BOW(object):
    def __init__(self, conf):
        self.vocab_size = conf["vocab_size"]
        self.emb_size = conf["emb_size"]
        self.hidden_size = conf.get("hidden_size", 128)
        self.n_class = conf["n_class"]
        if self.vocab_size and self.emb_size:
            self.emb_layer = EmbeddingLayer(self.vocab_size, self.emb_size)
        self.seq_pool_layer = SequencePoolingLayer()
        self.softsign_layer = SoftsignLayer()
        self.bow_layer = FCLayer(self.emb_size, self.hidden_size)
        self.fc_layer = FCLayer(2 * self.hidden_size, self.n_class)

    def net(self, tensor_dict):
        x_input = tensor_dict["input"]
        labels = tensor_dict["output"]
        keep_rate = tensor_dict["keep_rate"]
        assert isinstance(x_input, list)
        left, right = x_input[0:2]
        print("warning: only first vector is valid !!")
        if len(left.get_shape()) == 2:
            left_emb = self.emb_layer.ops(left)
            right_emb = self.emb_layer.ops(right)
        elif len(left.get_shape()) == 3:
            left_emb, right_emb = left, right
        left_pool = self.seq_pool_layer.ops(left_emb)
        right_pool = self.seq_pool_layer.ops(right_emb)
        left_soft = self.softsign_layer.ops(left_pool)
        right_soft = self.softsign_layer.ops(right_pool)
        left_bow = self.bow_layer.ops(left_soft)
        right_bow = self.bow_layer.ops(right_soft)
        concat = tf.concat([left_bow, right_bow], -1)
        logits = self.fc_layer.ops(concat)
        probabilities = tf.nn.dropout(logits, keep_prob=keep_rate)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probabilities, labels=labels))
        return tf.nn.softmax(probabilities), labels, loss


class ABCNN(object):
    '''
    from https://github.com/galsang/ABCNN.git
    '''
    def __init__(self, conf):
        self.model_type = conf.get("model_type", "BCNN")
        self.vocab_size = conf["vocab_size"]
        self.emb_size = conf["emb_size"]
        self.seq_len = conf["seq_len"]
        self.kernel_num = conf.get("num_filters", 50)
        self.win_size = conf.get("window_size", 4)
        self.pool_size = conf.get("pool_size", 4)
        self.l2_reg = conf.get("l2_reg", 0.0004)
        self.num_layers = conf.get("num_layers", 2)
        self.n_class = conf.get("n_class", 2)

        if self.vocab_size and self.emb_size:
            self.emb_layer = EmbeddingLayer(self.vocab_size, self.emb_size)
        self.b_cnn_0 = WideCNN(l2_reg=self.l2_reg, filter_num=self.kernel_num, filter_size=self.win_size,
                               emb_dim=self.emb_size, reuse=False)
        self.b_cnn_1 = WideCNN(l2_reg=self.l2_reg, filter_num=self.kernel_num, filter_size=self.win_size,
                               emb_dim=self.emb_size, reuse=False, emb_filter_size=self.kernel_num)
        self.pool_avg = AvgPooling2d(input_dim=self.emb_size, input_len=self.seq_len, pool_size=self.pool_size)
        self.all_pool0 = AllPooling2d(input_dim=self.emb_size, pool_size=self.seq_len)
        self.all_pool1 = AllPooling2d(input_dim=self.kernel_num, pool_size=self.seq_len+self.pool_size-1)

        self.out_features = None

    @staticmethod
    def cos_sim(v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")
        return dot_products / (norm1 * norm2)

    @staticmethod
    def make_attention_mat(x1, x2):
            # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]  dim, seq_len
            # x2 => [batch, height, 1, width]  [batch, d, 1, s]
            # [batch, width, wdith] = [batch, s, s]
            # matrix_transpose解决的是最后两位维度的顺序, 欧式距离切换l1，解决loss的nan问题（cos_sim造成）
            # euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            euclidean = tf.reduce_sum(tf.abs(x1 - tf.matrix_transpose(x2)), axis=1)
            return 1 / (1 + euclidean)

    def bcnn_block(self, variable_scope, x1, x2, input_emb_size):
        # x1, x2 = [batch, d, s, 1]
        with tf.variable_scope(variable_scope):
            if self.model_type == "ABCNN1" or self.model_type == "ABCNN3":
                with tf.name_scope("att_mat"):
                    aW = tf.get_variable(name="aW", shape=(self.seq_len, input_emb_size),
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
                    att_mat = self.make_attention_mat(x1, x2) # 4个维度, 消除emb的维度
                    # [batch, s, s] * [s,d] => [batch, s, d], transpose => [batch, d, s]
                    x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                    x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl",
                                                                        tf.matrix_transpose(att_mat), aW)), -1)
                    # 得到与输入同纬度的权重矩阵，[batch, d, s, 2], 权重作为通道层
                    x1 = tf.concat([x1, x1_a], axis=3)
                    x2 = tf.concat([x2, x2_a], axis=3)
            if input_emb_size == self.emb_size:
                left_conv, right_conv = self.b_cnn_0.ops(emb=x1), self.b_cnn_0.ops(emb=x2)
            else:
                left_conv, right_conv = self.b_cnn_1.ops(emb=x1), self.b_cnn_1.ops(emb=x2)

            left_attention, right_attention = None, None

            if self.model_type == "ABCNN2" or self.model_type == "ABCNN3":
                # [batch, s+w-1, s+w-1]
                att_mat = self.make_attention_mat(left_conv, right_conv)
                # [batch, s+w-1], [batch, s+w-1]
                left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

            left_wp = self.pool_avg.ops(input_conv=left_conv, attention=left_attention)
            left_ap = self.all_pool1.ops(input_conv=left_conv)
            right_wp = self.pool_avg.ops(input_conv=right_conv, attention=left_attention)
            right_ap = self.all_pool1.ops(input_conv=right_conv)
            return left_wp, left_ap, right_wp, right_ap

    def net(self, tensor_dict):
        x_input = tensor_dict["input"]
        labels = tensor_dict["output"]
        keep_rate = tensor_dict["keep_rate"]
        assert isinstance(x_input, list)
        print("warning: only first vector is valid !!")
        left, right = x_input[0:2]
        if len(left.get_shape()) == 2:
            left_emb = self.emb_layer.ops(left)
            right_emb = self.emb_layer.ops(right)
        elif len(left.get_shape()) == 3:
            left_emb, right_emb = left, right
        left_emb = tf.transpose(left_emb, [0, 2, 1])
        right_emb = tf.transpose(right_emb, [0, 2, 1])
        x1_expanded = tf.expand_dims(left_emb, -1)
        x2_expanded = tf.expand_dims(right_emb, -1)
        print("input expand: ", x1_expanded.get_shape())

        v0_left = self.all_pool0.ops(input_conv=x1_expanded)
        v0_right = self.all_pool0.ops(input_conv=x2_expanded)
        sims = [self.cos_sim(v0_left, v0_right)]
        print("input embedding all pool: ", v0_left.get_shape())

        conv1_left, v1_left, conv1_right, v1_right = self.bcnn_block(variable_scope="CNN-1",
                                                                     x1=x1_expanded, x2=x2_expanded,
                                                                     input_emb_size=self.emb_size)
        sims.append(self.cos_sim(v1_left, v1_right))

        if self.num_layers > 1:
            print("  CNN-2 input: ", conv1_left.get_shape())
            _, v2_left, _, v2_right = self.bcnn_block(variable_scope="CNN-2",
                                                      x1=conv1_left, x2=conv1_right, input_emb_size=self.kernel_num)
            sims.append(self.cos_sim(v2_left, v2_right))

        self.out_features = tf.concat([tf.stack(sims, axis=1)], axis=1, name="output_features")
        print("features shape: ", self.out_features.get_shape())

        estimation = tf.contrib.layers.fully_connected(
            inputs=self.out_features, num_outputs=self.n_class, activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
            biases_initializer=tf.constant_initializer(1e-04)
        )
        print("ouput of network: ", estimation.get_shape())

        probabilities = estimation # tf.nn.dropoutkeep_prob=keep_rate)
        label_ix = tf.cast(tf.argmax(labels, 1), tf.int32)
        loss = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=probabilities, labels=label_ix)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
        return tf.nn.softmax(probabilities), labels, loss


nets_dict = {"textcnn": TextCnn, "bert": BertClassifier, "bow": BOW, "dssm": DSSM, "abcnn": ABCNN}
