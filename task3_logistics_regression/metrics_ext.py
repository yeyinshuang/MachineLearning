# -*- coding: utf-8 -*-

###Task 1 完成sigmoid函数的定义
##输入变量一个/多个数值，返回值为经历sigmoid函数之后的结果


import numpy as np

def Sigmoid_fn(x1):

    # sigmoid = 1/(1+e^（-x1))
    sigmoid = 1/(1+np.exp(-x1))

    #你的定义

    return sigmoid

print(Sigmoid_fn(2))
x = np.array([1,2,3,4,5,10,50])
print(Sigmoid_fn(x))
