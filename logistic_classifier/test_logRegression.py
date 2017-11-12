# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/8 17:31
 @Author: sunxiang
"""
from numpy import *
import matplotlib.pyplot as plt
import time

from logRegression import *


def loadData():
    train_x = []
    train_y = []
    # fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')
    fileIn = open('testSet.txt')

    # fileIn = open('data_3121867.txt')
    # fileIn = open('test_dian.txt')
    # fileIn = open('data_dianshang_2.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])

        if lineArr[2] == 1 or lineArr[2] == '1':
            train_y.append(float(0))
        else:
            train_y.append(float(1))
    return mat(train_x), mat(train_y).transpose()


dataArr, labelMat = loadData()
print grandAscent(dataArr, labelMat)


# ## step 1: load data
# print "step 1: load data..."
# train_x, train_y = loadData()
# test_x = train_x
# test_y = train_y
#
# ## step 2: training...
# print "step 2: training..."
# opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
# optimalWeights = trainLogRegres(train_x, train_y, opts)
#
# ## step 3: testing
# print "step 3: testing..."
# accuracy = testLogRegres(optimalWeights, test_x, test_y)
#
# ## step 4: show the result
# print "step 4: show the result..."
# print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
# showLogRegres(optimalWeights, train_x, train_y)
