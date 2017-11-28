#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy
import math
import matplotlib.pyplot as plt

###
 # 加载数据集
 # X -> [[x1],[x2],[x3],...,[xn]]
 # Y -> [y1, y2, y3, ..., yn]
####
def loadDataSet():

    dataMat = numpy.array([])
    labelMat = numpy.array([])

    fr = open(name='testSet.txt', mode='r')

    for line in fr :
        vstr = line.strip().split('\t')

        numMat = numpy.array([float(vstr[0]), float(vstr[1])]).reshape(2,1)
        label = int(vstr[2])

        if(dataMat.shape[0] == 0):
            dataMat = numMat
        else:
            dataMat = numpy.hstack((dataMat, numMat))

        if labelMat.shape[0] == 0:
            labelMat = numpy.array([label])
        else:
            labelMat = numpy.hstack((labelMat, label))

    return dataMat, labelMat

###
# 求 sigmoid 函数
##
def sigoid(inX):
    return 1.0 / (1+math.exp(-inX))

##
# 求梯度的过程 ：
#               Y = w*X
##
def gradAscent(dataMat, labelMat):

    # 原始的参数 w 置为 1
    wMat = numpy.ones((2,1))

    for i in range(dataMat.shape[1]):
        # 将每一列的值带入计算， 计算出结果
        z = numpy.dot(dataMat[:, i],wMat)
        res = sigoid(z)
        # 步条
        alpha=0.01
        ## 计算 dY,
        dY = labelMat[i] - res
        # w = w+dw(alpha * X * dy)
        wMat = wMat + alpha * dataMat[:,i].reshape((2,1)) * dY

        print(res, labelMat[i], wMat)


##
# 随机梯度上升法
##

def stocGradAscent0(dataMat, labelMat):
    m,n = dataMat.shape
    alpha = 0.01

    w = ones(m)

    for i in range(n):
        z = sigoid()

def main():
    dataMat, labelMat = loadDataSet()

    gradAscent(dataMat, labelMat)

if __name__ == '__main__':
    main()
