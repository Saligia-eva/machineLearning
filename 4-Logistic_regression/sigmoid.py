#!/usr/bin/env python
# -*- coding:utf8 -*-

import matplotlib.pyplot as plt
import numpy as np

# sigmoid 函数
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


# 三维绘图
def  gradientFunc():
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-2, 2, 0.1)
    Y = np.arange(-2, 2, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = X**2 + Y**2
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    x= np.linspace(1,2,10)
    y= np.linspace(2,4,10)

    plt.plot(x,y)
    
    plt.show()

def main():
    gradientFunc()

if __name__ == '__main__':
    main()
