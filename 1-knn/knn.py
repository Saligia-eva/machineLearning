#!/usr/bin/env python
#-*- coding:utf8 -*-

import numpy
import numpy as np

# [X1, X2, X3, ..., Xn] -> [y1, y2, y3, ... yn]
# X 为向量矩阵， y 为标签值
# 向量矩阵跟向量对应的标签横向扩展，方便计算

def createDataset():
    dataset=numpy.array([[1,2], [2,3], [3,2], [-1,2],[-2,2],[-2,1]]).T
    titles=numpy.array(['A','A','A','B','B','B'])

    return dataset,titles

def knnGetTitle(inX, dataset, titles, k):

    """
    首先计算出输入向量与所有向量之间的距离
    然后按照距离排序
    从距离最近的前 k 个标签中，找到出现的频率最高的标签来
    """

    metricSet=dataset-inX  # 向量广播求差集
    metricSet=metricSet**2
    metricSet=metricSet.sum(axis=0)
    metricSet=metricSet**0.5


    sortArg=metricSet.argsort()

    argCount={}   # title:count

    for i in range(k):
        key=titles[sortArg[i]]
        argCount[key] = argCount.get(key, 0) +1

    sortCount=sorted(argCount.items(), key=lambda b:b[1], reverse=True)

    return sortCount[0][0]

def main():
    X=numpy.array([3,4])
    X=X.reshape(X.shape[0],1)

    dataset,titles=createDataset()
    res = knnGetTitle(inX=X, dataset=dataset, titles=titles, k=4)
    print(res)
    dataset=np.hstack((dataset, X))
    titles=np.hstack((titles, res))

    print(dataset,titles)

if __name__ == '__main__':
    main()
