#!/usr/bin/env python
# -*- coding:utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

def sigmoid():
    x = np.linspace(-8,8,1000)
    y = [1/(1+np.exp(-i)) for i in x]

    # 设置坐标
    plt.xlim((-10, 10))
    plt.ylim((0, 1))

    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_position(('data',0))

    plt.plot(x,y)
    plt.show()


def main():
    sigmoid()

if __name__ == '__main__':
    main()
