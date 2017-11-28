#!/usr/bin/env python
# -*- coding:utf8 -*-

import re
import random
import numpy

def textParse(bigString):

    listItem = re.split(r'\W', bigString)
    listItem = [vItem.lower() for vItem in listItem if len(vItem) > 2]
    #print(listItem)
    return listItem

def createVocabList(dataset):
    dicset = set([])

    for lineSet in dataset:
        dicset = dicset | set(lineSet)
    return list(dicset)

# 获取词频
def setofWords2Vec(vocabList, docList):
    dimVec = numpy.zeros((len(vocabList), 1))
    #print(dimVec)
    #dimVec = [0] * len(vocabList)  # 获取词量

    for word in docList:
        if word in vocabList :
            dimVec[vocabList.index(word),0] = 1
        else:
            print("word [%s] is out of dict" % word)

    return dimVec


# 获取概率
def trainNB0(trainMat, trainClasses):
    """
    计算总数
    分类求和
    求词概率
    """
    dimSize = trainMat.shape[0]      # 求词典维度

    p0Num = 0
    p0Vec = numpy.ones((dimSize, 1))
    p1Num = 0
    p1Vec = numpy.ones((dimSize, 1))

    for i in range(trainMat.shape[1]):
        if trainClasses[i] == 0:
            p0Vec += trainMat[:, i].reshape(dimSize,1)
            p0Num += 1
        else:
            p1Vec += trainMat[:, i].reshape(dimSize,1)
            p1Num += 1

    p0Vec = p0Vec/p0Num
    p1Vec = p1Vec/p1Num

    return p0Vec, p1Vec


# 利用概率求取分类
def classifyNB(testVec, p0Vec, p1Vec):

    p0 = numpy.sum(testVec*p0Vec)
    p1 = numpy.sum(testVec*p1Vec)

    if p0 > p1:
        return 0
    else :
        return 1


def spamTest():

    docList=[]
    classList = []
    fullText = []

    for i in range(1,26):

        fStr = open('email/spam/%d.txt' % i).read()
        fList = textParse(fStr)
        docList.append(fList)
        fullText.extend(fList)  # ?
        classList.append(1)

        fStr = open('email/ham/%d.txt' % i).read()
        fList = textParse(fStr)
        docList.append(fList)
        fullText.extend(fList)  # ?
        classList.append(0)

    # 构建词库
    vocabList = createVocabList(dataset=docList)

    trainSet = range(50)
    testSet = []

    for i in range(10):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del trainSet[randIndex]

    # 构建分类
    trainMat = numpy.array([])
    trainClasses = numpy.array([])


    for docIndex in trainSet :

        if trainMat.shape[0] == 0:
            trainMat = setofWords2Vec(vocabList, docList[docIndex])
        else:
            dictVec = setofWords2Vec(vocabList, docList[docIndex])
            trainMat = numpy.hstack((trainMat, dictVec))

        if trainClasses.shape[0] == 0:
            trainClasses = numpy.array([classList[docIndex]])
        else :
            trainClasses = numpy.hstack((trainClasses, classList[docIndex]))


    p0Vec, p1Vec = trainNB0(trainMat=trainMat, trainClasses=trainClasses)

    ## 构建测试

    testMat = numpy.array([])
    testClasses = numpy.array([])


    for docIndex in testSet :
        testMat = setofWords2Vec(vocabList, docList[docIndex])
        label = classList[docIndex]

        res = classifyNB(testMat, p0Vec=p0Vec, p1Vec=p1Vec)

        if label == res :
            print("SUCESS")
        else :
            print("ERROR")



def main():
    spamTest()

if __name__ == '__main__':
    main()
