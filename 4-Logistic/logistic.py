#!/usr/bin/env python

import numpy
import math

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

def sigoid(inX):
    return 1.0 / (1+math.exp(-inX))

def gradAscent(dataMat, labelMat):

    wMat = numpy.ones((2,1))

    for i in range(dataMat.shape[1]):
        z = numpy.dot(dataMat[:, i],wMat)
        res = sigoid(z)

        alpha=0.01

        dw = labelMat[i] - res

        wMat = wMat + alpha * dataMat[:,i].reshape((2,1)) * dw


        print(res, labelMat[i], wMat)

def main():
    dataMat, labelMat = loadDataSet()

    gradAscent(dataMat, labelMat)

if __name__ == '__main__':
    main()
