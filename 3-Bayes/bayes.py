#!/usr/bin/evn python
# -*- coding:utf8 -*-

################################
# 构建单词向量:
#      0 : 情感类评论
#      1 : 中性言论
#################################

import numpy
import math

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


def createVocabList(dataSet):
    """
    获取　dataset 下的所有元素集合
    """
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    # 构建 0 向量矩阵
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

####
# 计算 p(w|ci)
#
#    ci : 表示不同类别的概率
#
#    w : 代表词汇概率
#       如果每个词汇是相互独立的， 那么 p(w|ci) = p(w1|ci)*p(w2|ci) * ... * p(wn|ci) [保留]
#
#
####
def trainNB0(trainMatrix, trainCategory):

    numTraDocs=len(trainMatrix)   # 计算总的文档的数量
    numWords=len(trainMatrix[0])  # 计算总的词汇的数量

    pAbusive=trainCategory.count(1)/float(numTraDocs) # 计算情感性语言概率

    p0Num=numpy.zeros(numWords)
    p1Num=numpy.zeros(numWords)

    print(p0Num)
    print(p1Num)

    p0Denom=0.0
    p1Denom=0.0

    # 计算每个类别下每个的词频
    for i in range(numTraDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 计算每个单词出现的概率
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass)
    p0 = sum(vec2Classify * p0Vec) + math.log(1- pClass)

    if p1 > 0:
        return 1
    else:
        return 0


def main():
    database,labels=loadDataSet()

    #收集所有的词汇
    myVocabList=createVocabList(database)
    #res = setOfWords2Vec(myVocabList, database[0])
    trainMat = []

    for postinDoc in database:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V,p1V,pAb = trainNB0(trainMat, labels)

    #testEntry = ['love', 'my', 'dalmation']
    testEntry = ['stupid', 'garbage']

    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))

    print(testEntry,'classified as : ' , classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    main()
