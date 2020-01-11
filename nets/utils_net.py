# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng

import tensorflow as tf
import numpy as np
import math
tf.set_random_seed(49999)

tf_dtypes = [tf.int32, tf.int64, tf.float32, tf.float64]
np_dtypes = [np.int32, np.int64, np.float32, np.float64]


def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)


class FCLayer(object):
    """
    a layer class: a fc layer implementation in tensorflow
    """
    def __init__(self, num_in, num_out):
        """
        init function
        """
        self.num_in = num_in
        self.num_out = num_out
        self.weight = tf.Variable(tf.random_normal([num_in, num_out]))
        self.bias = tf.Variable(tf.random_normal([num_out]))

    def ops(self, input_x):
        """
        operation
        """
        out_without_bias = tf.matmul(input_x, self.weight)
        output = tf.nn.bias_add(out_without_bias, self.bias)
        return output


class CosineLayer(object):
    """
    a layer class: cosine layer
    """
    def __init__(self):
        """
        init function
        """
        pass

    def ops(self, input_a, input_b):
        """
        operation
        """
        norm_a = tf.nn.l2_normalize(input_a, dim=1)
        norm_b = tf.nn.l2_normalize(input_b, dim=1)
        # multiply is dot-product
        cos_sim = tf.expand_dims(tf.reduce_sum(tf.multiply(norm_a, norm_b), 1), -1)
        return cos_sim


class SoftsignLayer(object):
    """
    a layer class: softsign Activation function
    """
    def __init__(self):
        """
        init function
        """
        pass

    def ops(self, input_x):
        """
        operation
        """
        return tf.nn.softsign(input_x)


class SequencePoolingLayer(object):
    """
    a layer class: A sequence pooling implementation in tensorflow
    """
    def __init__(self):
        """
        init function
        """
        pass

    def ops(self, emb):
        """
        operation
        """
        reduce_sum = tf.reduce_sum(emb, axis=1)
        return reduce_sum


class EmbeddingLayer(object):
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        init_scope = math.sqrt(6.0 / (vocab_size + emb_dim))
        emb_shape = [self.vocab_size, self.emb_dim]
        self.embedding = tf.Variable(tf.random_uniform(emb_shape, -init_scope, init_scope), trainable=True)

    def ops(self, input_x):
        return tf.nn.embedding_lookup(self.embedding, input_x)


class WideCNN(object):
    def __init__(self, filter_size, filter_num, l2_reg, emb_dim, reuse=True, activation_fn="tanh", trans=True,
                 emb_filter_size=None):
        self.w, self.l2_reg, self.d = filter_size, l2_reg, emb_dim
        self.di = filter_num
        self.reuse = reuse
        if activation_fn == "tanh":
            self.activation_fn = tf.nn.tanh
        elif activation_fn == "relu":
            self.activation_fn = tf.nn.relu
        self.transpose = trans
        self.emb_filter_size = emb_filter_size

    def pad_for_wide_conv(self, x):
        return tf.pad(x, np.array([[0, 0], [0, 0], [self.w-1, self.w-1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

    def ops(self, emb):
        assert len(emb.shape) in [3, 4]
        emb_expanded = tf.expand_dims(emb, -1) if len(emb.shape) == 3 else emb
        emb_pad = self.pad_for_wide_conv(emb_expanded)
        if self.emb_filter_size is None:
            kernel_size = (self.d, self.w)
        else:
            kernel_size = (self.emb_filter_size, self.w)
        conv = tf.contrib.layers.conv2d(inputs=emb_pad, num_outputs=self.di, kernel_size=kernel_size,
                                        stride=1, padding="VALID", activation_fn=self.activation_fn,
                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                                        biases_initializer=tf.constant_initializer(1e-04),
                                        reuse=self.reuse, trainable=True)
        # Weight: [filter_height, filter_width, in_channels, out_channels]
        # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]
        # [batch, di, s+w-1, 1]
        if self.transpose:
            return tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
        else:
            return conv


class AllPooling2d(object):
    def __init__(self, input_dim, pool_size, name="all_ap"):
        self.input_dim = input_dim
        self.pool_size = pool_size
        self.layer_name = name

    def ops(self, input_conv):
        all_ap = tf.layers.average_pooling2d(inputs=input_conv, pool_size=(1, self.pool_size), strides=1,
                                             padding="VALID", name=self.layer_name)
        all_ap_reshaped = tf.reshape(all_ap, [-1, self.input_dim])
        return all_ap_reshaped


class AvgPooling2d(object):
    def __init__(self, input_dim, input_len, pool_size, name="w_ap"):
        self.input_dim = input_dim
        self.input_len = input_len
        self.pool_size = pool_size
        self.layer_name = name

    def ops(self, input_conv, attention=None):
        # x: [batch, di, s+w-1, 1]
        # attention: [batch, s+w-1]
        if attention is not None:
            pools = []
            # [batch, s+w-1] => [batch, 1, s+w-1, 1]
            attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])
            for i in range(self.input_len):
              # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
              pools.append(tf.reduce_sum(
                input_conv[:, :, i:i + self.pool_size, :] * attention[:, :, i:i + self.pool_size, :], axis=2,
                keep_dims=True))
            # [batch, di, s, 1]
            w_ap = tf.concat(pools, axis=2, name=self.layer_name)
        else:
            w_ap = tf.layers.average_pooling2d(inputs=input_conv, pool_size=(1, self.pool_size), strides=1,
                                             padding="VALID", name=self.layer_name)
        return w_ap
