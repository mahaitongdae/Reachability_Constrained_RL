#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/18
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: test_tf_torch.py
# =====================================
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import torch
import numpy as np
import torch.nn as nn
import gym
import time
from mpi4py import MPI

def tfNet(input_dim, num_hidden_layers, num_hidden_units, output_dim, name=None):
    x_input = tf.keras.Input(shape=(input_dim,))
    h = x_input
    for i in range(num_hidden_layers):
        h = tf.keras.layers.Dense(units=num_hidden_units, activation='tanh')(h)

    outputs = Dense(output_dim, activation='linear', dtype=tf.float32)(h)

    network = tf.keras.Model(inputs=[x_input], outputs=[outputs])
    return network


class TFNet(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, output_dim, **kwargs):
        super(TFNet, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units, input_shape=(input_dim,), activation='tanh', dtype=tf.float32)
        self.hidden = Sequential(
            [Dense(num_hidden_units, activation='tanh', dtype=tf.float32) for _ in range(num_hidden_layers - 1)])
        self.outputs = Dense(output_dim, activation='linear', dtype=tf.float32)
        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.hidden(x)
        x = self.outputs(x)
        return x


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class TorchNet(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, output_dim):
        super(TorchNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, num_hidden_units, bias=True)
        self.hidden = []
        for _ in range(num_hidden_layers-1):
            self.hidden.append(nn.Linear(num_hidden_units, num_hidden_units, bias=True))
        self.out = nn.Linear(num_hidden_units, output_dim, bias=True)
        init_weights(self)

    def forward(self, state):
        x = self.linear1(state)
        x = torch.tanh(x)
        for layer in self.hidden:
            x = layer(x)
            x = torch.tanh(x)
        out = self.out(x)

        return out


tfnet = tfNet(3, 2, 256, 1, name='tfnet')
torchnet = TorchNet(3, 2, 256, 1)


# @tf.function
def tfnet_forward(obs):
    action = tfnet(obs)
    return action


def test_tfnet(batch_size=2048):
    env = gym.make('Pendulum-v0')
    obs = env.reset()
    done = 0
    for _ in range(batch_size):
        action = tfnet_forward(obs[np.newaxis, :])
        obs_tp1, reward, done, info = env.step(action.numpy()[0])
        obs = env.reset() if done else obs_tp1.copy()


def test_torchnet(batch_size=2048):
    env = gym.make('Pendulum-v0')
    obs = env.reset()
    done = 0
    for _ in range(batch_size):
        action = torchnet.forward(torch.tensor(obs[np.newaxis, :], dtype=torch.float32))
        obs_tp1, reward, done, info = env.step(action.detach().numpy()[0])
        obs = env.reset() if done else obs_tp1.copy()

def test_tf_torch():
    env = gym.make('Pendulum-v0')
    obs = env.reset()
    tfnet_forward(obs[np.newaxis, :])

    torch_start_time = time.time()
    test_torchnet(2048)
    print(time.time() - torch_start_time)

    tf_start_time = time.time()
    test_tfnet(2048)
    print(time.time() - tf_start_time)

def test_tf():
    @tf.function
    def inner_test(a):
        for _ in range(1000):
            b = tf.square(a)

    s = np.random.random((10000, 10000))
    inner_test(s)

    start_time = time.time()
    # s = tf.constant(s)
    inner_test(s)

    print(time.time() - start_time)


if __name__ == '__main__':
    test_tf()





