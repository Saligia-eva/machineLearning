#!/usr/bin/env python
#-*- coding:utf8 -*-

import random
import numpy
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    labelMat= []

    fr = open(fileName)

    for line in fr:
        lineArr = line.strip().split('\t')

        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 向量参数
        labelMat.append(float(lineArr[2]))                    # 标签

    # dataMat = numpy.mat(dataMat).reshape(2,-1)
    # labelMat = numpy.mat(labelMat)

    return dataMat, labelMat

##
#
# 回去 aj 的下标
#
##
def selectJrand(i,m):
    j = 1
    while j == i:
        j = int(random.uniform(0, m))
    return j

###
#
###
def clipAlpha(aj, H, L):

    if aj > H :
        aj = H
    if L > aj :
        aj = L

    return aj

##
# dataMatIn   : 数据集
# classLabels : 标签
# c           : 常数 c
# toler       : 容错率
# maxIter     : 取消前最大的循环次数
#
##
def smoSimple(dataMatrix, labelMat, C, toler, maxIter):
    b = 0;
    m,n = dataMatrix.shape
    # m 为记录数量
    # n 为维度个数

    # 构建 alpha 参数
    alphas = numpy.mat(numpy.zeros((n,1)))

    iter = 0

    while (iter < maxIter):
        alphaPairsChanged = 0

        for i in range(n):
            # label * (m_n * n_1[取出特定列--所有记录的特定维度]) -> 某个记录的所有维度的求值
            # 计算预测类别
            fXi = float(numpy.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b

            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions

            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas

def smoSimpleTest(dataMatIn, classLabels, C, toler, maxIter):

    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    # m 为记录数
    # n 为 维度数
    print("m=%d, n=%d\n" %(m,n))
    #print(alphas)
    iter = 0

    while (iter < maxIter):
        alphaPairsChanged = 0
        # 遍历每一个记录
        for i in range(1):
            # 求标签误差

            # ？？？
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions

            # 如果误差超过容忍，需要优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 获取另外一个下标 aj, 并计算误差
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])


                # ai 与 aj 的优化过程
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();

                # 查找最低与最高的范围

                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H:
                    print "L==H";
                    continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter

    return b,alphas

def main():
    dataMat, labelMat = loadDataSet('testSet.txt')

    ##smoSimple(dataMat, labelMat, 0.6, 0.001, 1)
    smoSimpleTest(dataMat, labelMat, 0.6, 0.001, 1)


if __name__ == '__main__':
    main()
