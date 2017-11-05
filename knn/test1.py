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

# A 1

dataSet, labels = kNN.createDataSet()
input = array([1, 0, 0, 0, 0, 0, 2, 48, 1])
K = 3
output = kNN.classify(input, dataSet, labels, K)
print("测试数据为:", input, "分类结果为：", output)