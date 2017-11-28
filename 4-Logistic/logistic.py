#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy
import math
import matplotlib.pyplot as plt

###
 # 加载数据集
 # X -> [[x1],[x2],[x3],...,[xn]]
 # Y -> [y1, y2, y3, ..., yn]
 #
 # Y = WX : dY/dw = X -> dw = dY/X
 #
####


def loadDataSet2():
    dataMat = numpy.array([])
    labelMat = numpy.array([])

    fr = open(name='horseColicTraining.txt', mode='r')

    for line in fr :
        vstr = line.strip().split('\t')
        numMat = numpy.array([])

        for i in range(21):
            numMat = numpy.hstack((numMat, float(vstr[i])))

        numMat = numMat.reshape(21,1)

        label = int(float(vstr[21]))
        labelMat = numpy.hstack((labelMat, label))

        if(dataMat.shape[0] == 0):
            dataMat = numMat
        else:
             dataMat = numpy.hstack((dataMat, numMat))

    return dataMat, labelMat

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
def gradAscent(dataMat, labelMat, w=0):

    m,n = dataMat.shape
    # 原始的参数 w 置为 1
    wMat = numpy.ones((m,1))

    if isinstance(w, numpy.ndarray):
        wMat = w

    error=0
    for i in range(n):
        # 将每一列的值带入计算， 计算出结果
        z = numpy.dot(dataMat[:, i],wMat)

        res = sigoid(z)
        # 步条
        alpha=0.01
        ## 计算 dY,
        dY = labelMat[i] - res
        # w = w+dw(alpha * X * dy)
        # 如何友好迭代每一个值
        # for 表达式？
        wMat = wMat + alpha * dataMat[:,i].reshape((m,1)) * dY

        if int(0.5+res) !=  int(labelMat[i]):
            error=error+1

    print("res : %f" % (float(error)/n))

    return wMat

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
    dataMat, labelMat = loadDataSet2()
    print(type(labelMat))
    w = 0
    for i in range(1000):
        w = gradAscent(dataMat, labelMat, w)

if __name__ == '__main__':
    main()
