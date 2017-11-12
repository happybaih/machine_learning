# -*- coding: utf-8 -*-
"""
 @Time: 2017/9/26 16:32
 @Author: sunxiang
"""

from numpy import *
import operator
import numpy as np


def data_ready(filename, rankcout):
    data = []
    labels = []
    with open(filename) as ifile:
        for line in ifile:
            try:
                tokens = line.strip().split(' ')
                data.append([float(tk) for tk in tokens[:-1]])

                # 0 的数据去除 效果好点
                # flag = True
                # one = list()
                # for tk in tokens[:-1]:
                #     if int(tk) == 0 or tk == '0':
                #         flag = False
                #     one.append(float(tk))
                # if flag:
                #     data.append(one)

                labels.append(tokens[-1])
            except:
                print line
    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)

    ''''' 标签转换为0/1 '''
    # y[labels == '1'] = 1
    for i in range(1, rankcout + 1):
        y[labels == str(i)] = i
    return x, y


# 给出训练数据以及对应的类别
def createDataSet():
    # group = array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    # labels = ['A', 'A', 'B', 'B']
    # return group, labels
    return data_ready()


# 通过KNN进行分类
def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]
    # 计算欧式距离
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  # 行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    # print("dist", dist)
    # 对距离进行排序
    sortedDistIndex = argsort(dist)  # argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        # 对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes
