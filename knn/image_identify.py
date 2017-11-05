# -*- coding: utf-8 -*-
"""
 @Time: 2017/9/26 15:14
 @Author: sunxiang
"""

# Numpy包数组操作API格式化数据
file_folder = "F:/study_file/test_data/picture/"


def loadPicture():
    train_index = 0
    test_index = 0
    train_data = np.zeros((200, 171, 171))
    test_data = np.zeros((160, 171, 171))
    train_label = np.zeros((200))
    test_label = np.zeros((160))
    for i in np.arange(40):
        image = mpimg.imread(file_folder + 'picture/' + str(i) + '.tiff')
        data = np.zeros((513, 513))
        data[0:image.shape[0], 0:image.shape[1]] = image
        # 切割后的图像位于数据的位置
        index = 0
        # 将图片分割成九块
        for row in np.arange(3):
            for col in np.arange(3):
                if index < 5:
                    train_data[train_index, :, :] = data[171 * row:171 * (row + 1), 171 * col:171 * (col + 1)]
                    train_label[train_index] = i
                    train_index += 1
                else:
                    test_data[test_index, :, :] = data[171 * row:171 * (row + 1), 171 * col:171 * (col + 1)]
                    test_label[test_index] = i
                    test_index += 1
                index += 1
    return train_data, test_data, train_label, test_label


# 特征提取 LBP特征提取方法
radius = 1
n_point = radius * 8


def texture_detect():
    train_hist = np.zeros((200, 256))
    test_hist = np.zeros((160, 256))
    for i in np.arange(200):
        # 使用LBP方法提取图像的纹理特征.
        lbp = skft.local_binary_pattern(train_data[i], n_point, radius, 'default')
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist size:256
        train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    for i in np.arange(160):
        lbp = skft.local_binary_pattern(test_data[i], n_point, radius, 'default')
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist size:256
        test_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    return train_hist, test_hist


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft

train_data, test_data, train_label, test_label = loadPicture()
train_hist, test_hist = texture_detect()
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
OneVsRestClassifier(svr_rbf, -1).fit(train_hist, train_label).score(test_hist, test_label)
