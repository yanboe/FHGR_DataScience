#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def gen_tensor_one_feature():
    space_index = np.linspace(-1, 1, 100)
    bell_curve = norm.pdf(space_index, 0, 0.5)
    case1 = np.repeat(bell_curve, 100).reshape(100, 100)
    case2 = np.repeat(bell_curve, 100).reshape(100, 100)
    case3 = np.repeat(bell_curve, 100).reshape(100, 100)
    case2[:, 50:] = case2[:, 50:] + 0.1
    case3[:, 50:] = case3[:, 50:] - 0.1
    X = np.zeros(90*100*100).reshape(90,100,100)
    for i in range(30):
        X[i,:,:] = case1 + np.random.normal(0,0.01,10000).reshape(100, 100)
        X[i+30,:,:] = case2 + np.random.normal(0,0.01,10000).reshape(100, 100)
        X[i+60,:,:] = case3 + np.random.normal(0,0.01,10000).reshape(100, 100)
    return X


def gen_tensor_three_feature():
    space_index = np.linspace(-1, 1, 100)
    bell_curve1 = norm.pdf(space_index, 0, 0.5)
    bell_curve2 = norm.pdf(space_index, -0.5, 0.5)
    bell_curve3 = norm.pdf(space_index, 0.5, 0.5)
    case1 = np.repeat(bell_curve1, 100).reshape(100, 100)
    case2 = np.repeat(bell_curve2, 100).reshape(100, 100)
    case3 = np.repeat(bell_curve3, 100).reshape(100, 100)
    space_index = np.linspace(0, 8*np.pi, 100)
    sine_wave = 0.2*np.sin(space_index)
    case1 = case1*sine_wave
    case2 = case2*sine_wave
    case3 = case3*sine_wave
    case2[:, 50:] = case2[:, 50:] + 0.1
    case3[:, 50:] = case3[:, 50:] - 0.1
    X = np.zeros(90*100*100).reshape(90,100,100)
    for i in range(30):
        X[i,:,:] = case1 + np.random.normal(0,0.01,10000).reshape(100, 100)
        X[i+30,:,:] = case2 + np.random.normal(0,0.01,10000).reshape(100, 100)
        X[i+60,:,:] = case3 + np.random.normal(0,0.01,10000).reshape(100, 100)
    return X


def plot_uvw_one_feature(data):
    label = ['sample', 'space', 'time']
    title = ['u[0]', 'v[0]', 'w[0]']
    style = ['r-', 'b-', 'g-']
    fig, axs = plt.subplots(3)
    for i in range(3):
        axs[i].set_ylabel(label[i])
        axs[i].set_title(title[i])
        axs[i].get_yaxis().set_ticks([])
        axs[i].plot(np.linspace(1, np.size(data[i]), np.size(data[i])), data[i], style[i])
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def plot_uvw_three_feature(data):
    label = ['sample', 'space', 'time']
    title = ['u[', 'v[', 'w[']
    style = ['r-', 'b-', 'g-']
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            if j == 0: axs[i, j].set_ylabel(label[i])
            axs[i, j].set_title(title[i] + str(j) + ']')
            axs[i, j].get_yaxis().set_ticks([])
            axs[i, j].plot(np.linspace(1, np.size(data[i][:,j]), np.size(data[i][:,j])), data[i][:,j], style[i])
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()


def plot_faces(data, x=3, y=6, rnd=True):
    if rnd == True:
        idx = np.absolute(1000*np.random.normal(0, 1, x*y)).astype(int)%400
    else:
        idx = np.arange(x*y)
    fig, axs = plt.subplots(x, y)
    for i, ax in enumerate(axs.flat):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(data[idx[i]], cmap='gray')
    plt.show()

