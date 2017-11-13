#!/usr/bin/env python
#-*- coding:utf8 -*-


##########################################################################################
#  决策树:
#    -> 特征划分
#    -> 信息熵
#
#　数据分类特点 :
#    特征　: [no sufacing(不浮出水面是否可以生存), flippers(是否有脚蹼)] -> flag(是否属于鱼类)
#
##############################################################################################

import math
import numpy
import operator

# 测试数据集
def createDataset():
    dataset=[[1,1,'yes'], [1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing', 'flippers']

    return dataset, labels

###
# 信息熵的计算公式：
#　　　xl = p(l)log(pl)
###
def calcShannonEnt(dataset):
    numEntries=len(dataset)

    labelCounts={}

    # 计算标签频率
    for vec in dataset:
        currentLabel=vec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0)+1

    shanonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 计算标签概率

        shanonEnt += (-prob*math.log(prob, 2))
    print("%s [%f]" % (labelCounts,shanonEnt))
    return shanonEnt

###########################################################
#   划分数据集
#       dataset : 数据集
#       axis    : 选取的特征
#       value   : 筛选特征值
#
#       return  : 返回的是只有这个特征值的扣除这一特征的矩阵
#
###########################################################
def splitDataset(dataset, axis, value):
    retDataset=[]
    numEntries=len(dataset)

    for vec in dataset:
        if vec[axis] == value :
            reduceFeatVect = vec[:axis]
            reduceFeatVect.extend(vec[axis+1:])

            retDataset.append(reduceFeatVect)

    return retDataset

# 选择最好的数据集划分方式

def chooseBestFeatureToSplit(dataset):
    numFeature=len(dataset[0])-1                ## 特征数量
    baseEntropy=calcShannonEnt(dataset)         ## 计算基础的信息熵

    bestInfoGain=0.0 # 比较熵
    bestFeature = -1 # 比较特征

    # 遍历特征集
    for i in range(numFeature):
        print("feat[%d]" % i)
        # 穷举特征集下面的所有特征值
        featList=[example[i] for example in dataset]
        uniqueVals=set(featList)

        newEntropy=0.0
        for value in uniqueVals:
            # 分离出某个特征下面对应某个特征值的某个子矩阵
            subDataset=splitDataset(dataset,i, value)

            # 取到这个子矩阵的概率
            prob=len(subDataset)/float(len(dataset))

            # 得到这个子矩阵的信息熵
            ent=prob*calcShannonEnt(subDataset)

            newEntropy+=ent
            #print(newEntropy)
        print("newEntropy : [%f]" % newEntropy)

        # 计算信息增益[熵减程度] 获得最优的信息增益情况
        infoGain=baseEntropy-newEntropy

        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

def majorityCnt(classList):
    classCount={}

    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) +1

    sortedClassCount=sorted(classCount.items(), key=lambda key:key[1], reversed=True)

    print(sortedClassCount)
    return sortedClassCount[0][0]


## 创建决策树
def createTree(dataset, labels):
    # 获取结果标签
    classList=[example[-1] for example in dataset]

    # 判断是否是确定信息
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # ????
    if len(dataset[0]) == 0:
        return majorityCnt(classList)

    # 从　dataset 中选择　bestFeat 特征作为分割
    bestFeat=chooseBestFeatureToSplit(dataset)
    bestFeatLabel=labels[bestFeat]

    myTree={bestFeatLabel:{}}

    # 对应于矩阵的坍缩，　所以标签也需要删除
    del(labels[bestFeat])

    featValues=[example[bestFeat] for example in dataset] # 选择所有的特征集
    uniqueVals=set(featValues)

    for value in uniqueVals:
        subLabels=labels[:]
        childData=splitDataset(dataset, bestFeat, value) # 获取到子集信息
        myTree[bestFeatLabel][value] = createTree(childData, subLabels)

    return myTree



def main():
    dataset,labels=createDataset()

    myTree=createTree(dataset, labels)

    print(myTree)

if __name__ == '__main__':
    main()
