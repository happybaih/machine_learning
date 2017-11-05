# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/2 13:31
 @Author: sunxiang
"""
from math import log
import numpy as np


# 度量数据集的无序程度

def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 为所有可能的分类创建字典
        currrentLabel = featVec[-1]
        if currrentLabel not in labelCounts.keys():
            labelCounts[currrentLabel] = 0
        labelCounts[currrentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 以2 为底求对数
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def data_ready():
    """
    数据读入  已有分类标签
    """
    data = []
    labels = []
    with open("data.txt") as ifile:
        for line in ifile:
            try:
                tokens = line.strip().split(' ')
                data.append([float(tk) for tk in tokens[:-1]])
                labels.append(tokens[-1])
            except:
                print line
    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)

    ''''' 标签转换为0/1 '''
    y[labels == 'A'] = 1
    return x, y


def data_ready_notype():
    """
    数据读入  已有分类标签 返回列表
    """
    data = []
    labels = []
    with open("data.txt") as ifile:
        for line in ifile:
            try:
                tokens = line.strip().split(' ')
                data.append([tk for tk in tokens])
                labels.append(tokens[-1])
            except:
                print line
    # x = np.array(data)
    # labels = np.array(labels)
    # y = np.zeros(labels.shape)
    #
    # ''''' 标签转换为0/1 '''
    # y[labels == 'A'] = 1
    return data


def createDataSet():
    """
    创建数据集
    :return:
    """
    return data_ready_notype()
    # dataSet = [[1, 1, 'yes'],
    #            [1, 1, 'yes'],
    #            [1, 0, 'no'],
    #            [0, 1, 'no'],
    #            [0, 1, 'no']]
    # labels = ['no surfacing', 'flippers']
    # return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet:  带划分数据集
    :param axis:    划分数据集的特征
    :param value:   特征的返回值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 发现符合要求的值 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# myDat, labels = createDataSet()
# myDat[0][-1] = 'maybe'      # 熵 越高则混合的数据越多
# myDat[-1][-1] = 'type'
# print calcShannonEnt(myDat)

# myDat, labels = createDataSet()
# print splitDataSet(myDat, 0, 1)  # 第一个为1 的  [[1, 'yes'], [1, 'yes'], [0, 'no']]
# print splitDataSet(myDat, 0, 0)  # 第一个为0 的  [[1, 'no'], [1, 'no']]

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    计算得出最好的划分数据集的特征
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 计算整个数据集的原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 计算两种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt((subDataSet))
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            # 计算最好的信息增益  索引值
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 哪一个特征最好的划分数据集
# myDat = createDataSet()
# print chooseBestFeatureToSplit(myDat)
import operator


def majorityCnt(classList):
    """
    排序字典 返回出现次数最多的分类名称
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创建树
    :param dataSet: 数据集
    :param labels:  标签列表
    :return:
    """
    classList = [example[-1] for example in dataSet]
    # 1、类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 2 遍历完所有特征时返回次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 3 得到列表中包含的所有属性值
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
