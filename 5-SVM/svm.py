#!/usr/bin/env python
#-*- coding:utf8 -*-

import random
import numpy

##
# 加载数据集合
##
def loadDataSet(fileName):
    dataMat = []
    labelMat= []

    fr = open(fileName)

    for line in fr:
        lineArr = line.strip().split('\t')

        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 向量参数
        labelMat.append(float(lineArr[2]))                      # 标签

    dataMat = numpy.mat(dataMat).reshape(2,-1)
    labelMat = numpy.mat(labelMat)

    return dataMat, labelMat

##
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

"""
创建 alpha 向量并将其初始化为 0 向量
当迭代次数小于最大迭代次数时(外循环)
   对数据集中的每个数据向量(内向量)
    如果该数据向量可以被优化：
     随机选择另外一个数据向量
     同时优化这两个数据向量
     如果两个向量都不能被优化,退出循环
如果所有的数据向量都没被优化，增加迭代次数，继续下一次循环
"""
##    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0

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
    """
def main():
    dataMat, labelMat = loadDataSet('testSet.txt')

    #print(labelMat)
    smoSimple(dataMat, labelMat, 0.6, 0.001, 40)

if __name__ == '__main__':
    main()
