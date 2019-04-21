# -*- coding: utf-8 -*-

###Task 1 完成距离度量函数的定义
##输入变量为两个向量，返回值为距离，不允许使用除数据结构外的第三方库

import numpy as np
from numpy import *

x1 = array([1,2,3])
x2 = array([4,5,6])

#曼哈顿距离
def distance_manhattan(x1,x2):
    distance = sum(abs(x1-x2))
    return distance


#欧式距离
def distance_euclidean(x1,x2):
    distance = sqrt(sum((x1-x2)**2))
    return distance


#余弦相似度
def distance_cosine(x1,x2):
     #cos_x1x2 = [x1,x2] / ||x1||*||x2|| 向量的内积比上向量的范数乘积

    v_dot = dot(x1,x2) #向量内积
    v_norm = np.linalg.norm(x1) * np.linalg.norm(x2) #范数
    distance = v_dot/v_norm
    return distance

# 测试曼哈顿距离
print(distance_manhattan(x1,x2))
#测试欧氏距离
print(distance_euclidean(x1,x2))
#测试余弦相似度
print(distance_cosine(x1,x2))


