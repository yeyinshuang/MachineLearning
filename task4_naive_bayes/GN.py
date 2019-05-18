# -*- coding: utf-8 -*-


###Task 手动构建贝叶斯的一些关键功能


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import math

data = pd.read_csv('vehicle.csv')

#以下返回值均为两个类别的对应，也就是说返回值应该为一个包含两个值的向量
#以下定义是按照计算层级定义的，你也可以自己定义自己的函数，直接顶一个包含很多函数的类去计算

feature_car = data[data['label']=='car'].iloc[:, [0, 1]].values
feature_truck = data[data['label']=='truck'].iloc[:, [0, 1]].values
# print(feature_car)

labels = data['label']
feature = data.drop('label',axis=1)


feature_mean = np.mean(feature, axis=0)
feature_mean_car = np.mean(feature_car, axis=0)
feature_mean_truck = np.mean(feature_truck, axis=0)

feature_std = np.std(feature, ddof=1, axis=0)
feature_car_std = np.std(feature_car, ddof=1, axis=0)
feature_truck_std = np.std(feature_truck, ddof=1, axis=0)





# print(feature)

from sklearn.model_selection import train_test_split
feature_train,featrre_test,label_train,label_test = train_test_split(data,labels,test_size=0.2,random_state=1)



# 计算特征数据的均值，
def get_mean(data_in):
	mean = np.mean(data_in)
	return mean

# 计算特征数据的标准差
def get_std(data_in):
	std = np.std(data_in)
	return std


# 利用上面的函数创建卡车和小轿车的高斯分布的均值和标准差，并计算出卡车和轿车的先验概率
# # 根据不同的交通工具，和以上计算的均值和标准差来计算输入数据的高斯概率

# μ代表均值  ，σ代表标准差
def gaussian_probability(data_in):

	# (1/np.sqrt(2*math.pi)*σ) exp(-((x-μ)**2 / 2*σ**2))

	mean = get_mean(data_in)
	std = get_std(data_in)
	gaussian_prob = (1/np.sqrt(2*np.pi)*std) * np.exp(-((data_in-mean)**2 / 2*(std**2)))

	return gaussian_prob


def gaussian_probability1(x):   # 变量x为包含长、宽两个特征值的向量
	gaussian_pro1=(1/(np.sqrt(np.pi*2)*get_std(x)))
	gaussian_pro2=np.exp((-np.square(x-get_mean(x))/2*np.square(get_std(x))))
	gaussian_pro=gaussian_pro1*gaussian_pro2
	return gaussian_pro
x=np.array([5,2])
y=gaussian_probability(x)
y1 = gaussian_probability1(x);
print(y)
print(y1)


#
# 用高斯概率，先验概率，贝叶斯公式计算输入特征的类别概率
# def class_probability(data_in):
#
# 	return class_prob
#
# # 比较输入特征的类别概率，来判断是car 还是truck
# def get_class(data_in):
#
# 	return class
#
# print(get_mean(data))
# print(get_std(data))
#
#
# print(gaussian_probability(data))








	

