# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/8 11:24
 @Author: sunxiang
"""
import numpy as np

filename = "data.csv"

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
    with open(filename) as ifile:
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


# def createDataSet():
#     """
#     创建数据集
#     :return:
#     """
#     return data_ready_notype()
#     # dataSet = [[1, 1, 'yes'],
#     #            [1, 1, 'yes'],
#     #            [1, 0, 'no'],
#     #            [0, 1, 'no'],
#     #            [0, 1, 'no']]
#     # labels = ['no surfacing', 'flippers']
#     # return dataSet, labels

def createDataSet():
    """
    创建数据集
    :return:
    """
    return np.loadtxt(filename, dtype=str, delimiter=",").tolist()

