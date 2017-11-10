#!/usr/bin/env python
#-*- coding:utf8 -*-


#########################
#    香弄熵计算公式
#########################

import math
import numpy

def createDataset():
    dataset=numpy.array([[1,1, 'test'], [1,1, 'yes'],[1,0,'heeh'],[0,1,'no'],[0,1, 'no']])

    return dataset.T

def calcShannonEnt(dataset):
    numEntries=len(dataset[1,:])

    labelCounts={}

    # 计算标签频率
    for i in range(numEntries):
        currentLabel=dataset[-1, i]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0)+1

    print(labelCounts)
    shanonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 计算标签概率
        print(prob)
        shanonEnt += (-prob*math.log(prob, 2))

    return shanonEnt

def main():
    dataset=createDataset()

    print(calcShannonEnt(dataset=dataset))

if __name__ == '__main__':
    main()
