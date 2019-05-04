# -*- coding: utf-8 -*-

###Task 1 完成评估函数的定义
##输入变量为预测值及真实值，返回值相应评估指标，只允许使用numpy和python标准库
#x1:课程中使用数据
#x2:课程中使用模型预测的预测值的真实值


# 回归的评估方法
from numpy import *

x1 = array([1,2,3,4,5])
x2 = array([1,2,3,4,5.1])#真实值

#绝对评价误差    sum(abs(x1-x2))/len(x1)
def metrics_MAE(x1,x2):
    mae = sum(abs(x2-x1))/len(x2)
    #你的定义
    return mae

#均方差sum((x1-x2) ** 2)/len(x1)
def metrics_MSE(x1,x2):
    mse = sum((x2-x1)**2)/len(x2)
    return mse

# R squared
def metrics_R2(x1,x2):

    mse = metrics_MSE(x1,x2)
    var = sum((x2-mean(x2))**2)/len(x2)
    r_squared = 1 - mse/var

    return r_squared

print(metrics_MAE(x1,x2))
print(metrics_MSE(x1,x2))
print((metrics_R2(x1,x2)))



