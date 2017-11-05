# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/3 14:19
 @Author: sunxiang
"""
import numpy as np
import urllib
# url with dataset
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# # download the file
# raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix

raw_data = ""
with open("data.txt", "r") as f:
    raw_data = f.readlines()
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]

# 数据标准化
from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)

# print X
# print "---------------"
# print y

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))