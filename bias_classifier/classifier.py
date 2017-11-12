# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/8 13:22
 @Author: sunxiang
"""
import csv


def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# filename = 'pima_indians_data.csv'
# dataset = loadCsv(filename)
# print('Loaded data file {0} with {1} rows').format(filename, len(dataset))


# 按类别划分数据
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]
# separated = separateByClass(dataset)
# print('Separated instances: {0}').format(separated)

import math


# 计算均值
def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# numbers = [1, 2, 3, 4, 5]
# print('Summary of {0}: mean={1}, stdev={2}').format(numbers, mean(numbers), stdev(numbers))

# 提取数据集的特征
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]
# summary = summarize(dataset)
# print('Attribute summaries: {0}').format(summary)

# 按类别提取属性特征
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries


# dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1], [4, 22, 0]]
# summary = summarizeByClass(dataset)
# print('Summary by class value: {0}').format(summary)

# 计算高斯概率密度函数
import math


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# x = 71.5
# mean = 73
# stdev = 6.2
# probability = calculateProbability(x, mean, stdev)
# print('Probability of belonging to this class: {0}').format(probability)

# 计算所属类的概率
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


# summaries = {0: [(1, 0.5)], 1: [(20, 5.0)]}
# inputVector = [1.1, '?']
# probabilities = calculateClassProbabilities(summaries, inputVector)
# print('Probabilities for each class: {0}').format(probabilities)


# 单一预测
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
# inputVector = [1.1, '?']
# result = predict(summaries, inputVector)
# print('Prediction: {0}').format(result)

# 多重预测
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
# testSet = [[1.1, '?'], [19.1, '?']]
# predictions = getPredictions(summaries, testSet)
# print('Predictions: {0}').format(predictions)

# 计算精度
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}').format(accuracy)
