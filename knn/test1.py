# -*- coding: utf-8 -*-
"""
 @Time: 2017/9/26 16:26
 @Author: sunxiang

 测试 knn 近邻算法
"""

import kNN
from numpy import *

# 1 0 0 69 0 174 2 186 2 B
# 2 22 5 0 0 0 0 0 0 B

# 2 0 0 0 0 0 2 13 1 A
# 2 0 0 3 0 0 2 65 1 A

# 1 0 0 0 0 0 2 48 1 A
# 1 0 0 0 0 0 2 582 0 A

# 1831 179 1

import random
import numpy as np


def splitDataset(dataset, splitRatio, labels):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    trainLabels = []

    copy = list(dataset)
    copyLabels = list(labels)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
        trainLabels.append(copyLabels.pop(index))
    return [trainSet, copy, trainLabels, copyLabels]

filename = "../data/3079066.txt"
# filename = "../data/data_3121867.txt"
rankcount = 199
dataSet, labels = kNN.data_ready(filename, rankcount)

# input = array([0, 1])
K = 5
rank_range = 10    # 排名误差
splitRatio = 0.67  # 训练集数据  测试集数据

trainingSet, testSet, trainLabels, copyLabels = splitDataset(dataSet.tolist(), splitRatio, labels.tolist())
print 'Split {0} rows into train={1} and test={2} rows'.format(len(dataSet), len(trainingSet), len(testSet))

success_count = 0

for i in range(0, len(testSet)):
    output = kNN.classify(testSet[i], np.array(trainingSet), np.array(trainLabels), K)
    copy_class = float(copyLabels[i])
    out_class = float(output)
    difference = int(abs(copy_class - out_class))
    # print difference
    if difference < rank_range:
        success_count += 1
    else:
        print testSet[i], output, copyLabels[i], difference
print success_count
print float(success_count)/len(testSet)
