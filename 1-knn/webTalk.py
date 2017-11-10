#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy
import numpy as np
import knn

def fileToDataset(dataFile):
    dataset=numpy.array([]);
    titles=[]

    fin=open(name=dataFile, mode="r")

    for v_line in fin :
        v_line = v_line.strip()
        v_item=v_line.split("\t")

        if len(v_item) is not 4 :
            continue

        vector=numpy.array([float(v_item[0]),float(v_item[1]),float(v_item[2])]).reshape(3,1)

        #print(vector)
        if dataset.shape[0] is 0:
            dataset = vector
        else :
            dataset = np.hstack((dataset, vector))

        titles = np.hstack((titles, v_item[3]))

    return dataset,titles

def autoNorma(dataSet):
    """
    归一化数值
    """
    KmSum=max(dataSet[0,:]) - min(dataSet[0,:])
    gameSum=max(dataSet[1,:]) - min(dataSet[1,:])
    moneySum=max(dataSet[2,:]) - min(dataSet[2,:])

    setSum=numpy.array([KmSum, gameSum, moneySum]).reshape(3,1)

    dataSet = dataSet/setSum

    return dataSet


def main():
    dataset,titles=fileToDataset('datingTestSet.txt')

    #dataset=autoNorma(dataset)

    m=len(dataset[1,:])

    testnum=int(m*(float(1)/5))

    metaset=dataset[:,:m-testnum]
    metatitle=titles[:m-testnum]


    testset=dataset[:,m-testnum:]
    testtitle=titles[m-testnum:]

    errorCount=0

    for i in range(testnum):

        inX=testset[:,i].reshape(3,1)

        res = knn.knnGetTitle(inX=inX,dataset=metaset,titles=metatitle, k=3)

        print("%d record test : res[%s] is %s" % (i, res, testtitle[i]))

        metaset = numpy.hstack((metaset, inX))
        metatitle=numpy.hstack((metatitle, res))

        print(metaset)
        print(metatitle)

        if res != testtitle[i]:
            errorCount=errorCount+1

    print("error count %d of %f" % (errorCount, float(errorCount)/m))

if __name__ == '__main__':
    main()
