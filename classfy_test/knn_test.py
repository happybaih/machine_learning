# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/2 10:08
 @Author: sunxiang
"""
from numpy import *
import numpy as np
import kNN


def data_ready():
    """ 数据读入
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


# 给出训练数据以及对应的类别
def createDataSet():
    # group = array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    # labels = ['A', 'A', 'B', 'B']
    # return group, labels

    return data_ready()


# 1 0 0 69 0 174 2 186 2 B
# 2 22 5 0 0 0 0 0 0 B

# 2 0 0 0 0 0 2 13 1 A
# 2 0 0 3 0 0 2 65 1 A

# 1 0 0 0 0 0 2 48 1 A
# 1 0 0 0 0 0 2 582 0 A

dataSet, labels = createDataSet()
input = array([1, 0, 0, 0, 0, 0, 2, 48, 1])
K = 3
output = kNN.classify(input, dataSet, labels, K)
print("测试数据为:", input, "分类结果为：", output)
