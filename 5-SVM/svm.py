#!/usr/bin/env python
#-*- coding:utf8 -*-

import random

def loadDataSet(fileName):
    dataMat = []
    labelMat= []

    fr = open(fileName)

    for line in fr:
        lineArr = line.strip().split('\t')

        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 向量参数
        labelMat.append(float(lineArr[2]))                    # 标签

    return dataMat, labelMat

##
#
# i : 第一个 alpha 的下标, m 是所有 alpha 的数据
#
##
def selectJrand(i,m):
    j = 1

    while j == i:
        j = int(random.uniform(0, m))

    return j

def cliAlpha(aj, H, L):

    if aj > H :
        aj = H
    if L > aj :
        aj = L

    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    



def main():
    dataMat, labelMat = loadDataSet('testSet.txt')

    print(labelMat)


if __name__ == '__main__':
    main()
