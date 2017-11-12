# -*- coding: utf-8 -*-
"""
 @Time: 2017/11/8 10:43
 @Author: sunxiang
"""

import numpy as np

# filename = "data_dian7.txt"

filename = "../data/data_3121867.txt"
data = []
labels = []
with open(filename) as ifile:
    for line in ifile:
        try:
            tokens = line.strip().split(' ')
            # data.append([tk for tk in tokens[:-1]])
            data.append([tk for tk in tokens])
            labels.append(tokens[-1])
        except:
            print line

filedata = np.array(data)

# print labels
#
# print filedata
np.savetxt('../data/dianshang_3121867.csv', filedata, delimiter=',', fmt='%s', newline='\n')

# print np.loadtxt('data.csv', dtype=str, delimiter=",").tolist()

