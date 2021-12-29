#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: model.py
# =====================================

import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import numpy as np

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class MLPNet(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units,
                            activation=hidden_activation,
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.1),
                            dtype=tf.float32)
        self.hidden = Sequential([Dense(num_hidden_units,
                                        activation=hidden_activation,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.1),
                                        dtype=tf.float32) for _ in range(num_hidden_layers-1)])
        output_activation = kwargs['output_activation'] if kwargs.get('output_activation') else 'linear'
        if kwargs.get('output_bias'):
            self.outputs = Dense(output_dim,
                                 activation=output_activation,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05),
                                 bias_initializer=tf.keras.initializers.Constant(kwargs.get('output_bias')),
                                 dtype=tf.float32)
        else:
            self.outputs = Dense(output_dim,
                                 activation=output_activation,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.05),
                                 bias_initializer=tf.keras.initializers.Constant(0.),
                                 dtype=tf.float32)
        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.hidden(x)
        x = self.outputs(x)
        return x


class AlphaModel(Model):
    def __init__(self, **kwargs):
        super(AlphaModel, self).__init__(name=kwargs['name'])
        self.log_alpha = tf.Variable(0., dtype=tf.float32)

class LamModel(Model):
    def __init__(self, **kwargs):
        super(LamModel, self).__init__(name=kwargs['name'])
        self.var = tf.Variable(-10., dtype=tf.float32)



def test_alpha():
    import numpy as np
    alpha_model = AlphaModel(name='alpha')
    print(alpha_model.trainable_weights)
    print(len(alpha_model.trainable_weights))
    print(alpha_model.get_weights())
    print(alpha_model.log_alpha)
    b = alpha_model.log_alpha
    alpha_model.set_weights(np.array([3]))
    print(b)

    with tf.GradientTape() as tape:
        b = 3.*alpha_model.log_alpha
    print(tape.gradient(b, alpha_model.trainable_weights[0]))

def test_lam():
    import numpy as np




    with tf.GradientTape() as tape:
        lam_model = LamModel(name='lam')
        print(lam_model.trainable_weights)
        print(len(lam_model.trainable_weights))
        print(lam_model.get_weights())
        print(lam_model.log_lam)
        b = lam_model.log_lam
        lam_model.set_weights(np.array([3]))
        print(b)
        c = 3.*lam_model.log_lam
    print(tape.gradient(c, lam_model.trainable_weights[0]))


def test_attrib():
    import numpy as np

    a = Variable(0, name='d')

    p = MLPNet(2, 2, 128, 1, name='ttt')
    print(hasattr(p, 'get_weights'))
    print(hasattr(p, 'trainable_weights'))
    print(hasattr(a, 'get_weights'))
    print(hasattr(a, 'trainable_weights'))
    print(type(a))
    print(type(p))
    # print(a.name)
    # print(p.name)
    # p.build((None, 2))
    p.summary()
    # inp = np.random.random([10, 2])
    # out = p.forward(inp)
    # print(p.get_weights())
    # print(p.trainable_weights)


def test_clone():
    p = MLPNet(2, 2, 128, 1, name='ttt')
    print(p._is_graph_network)
    s = tf.keras.models.clone_model(p)
    print(s)


def test_out():
    import numpy as np
    Qs = tuple(MLPNet(8, 2, 128, 1, name='Q' + str(i)) for i in range(2))
    inp = np.random.random((128, 8))
    out = [Q(inp) for Q in Qs]
    print(out)


def test_memory():
    import time
    Q = MLPNet(8, 2, 128, 1)
    time.sleep(111111)


def test_memory2():
    import time
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(30,), activation='relu'),
                                 tf.keras.layers.Dense(20, activation='relu'),
                                 tf.keras.layers.Dense(20, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='relu')])
    time.sleep(10000)


if __name__ == '__main__':
    test_lam()
