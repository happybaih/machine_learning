# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/9 10:29
 @Author: sunxiang
"""
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)

    # for line in fr.readlines():
    #     lineArr = line.strip().split('\t')
    #     dataMat.append([float(lineArr[0]), float(lineArr[1])])

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    """
    计算两个向量的欧式距离
    :param vecA:
    :param vecB:
    :return:
    """
    return sqrt(sum(power(vecA - vecB), 2))


def randCent(dataSet, k):
    """
    构建包含k个随机质心的集合
    :param dataSet:
    :param k:
    :return:
    """
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

filename = "testSet.txt"
dataMat = mat(loadDataSet(filename))

print dataMat[0][0]
print dataMat[1][0]
print distEclud(dataMat[0], dataMat[1])
# print randCent(dataMat, 2)

