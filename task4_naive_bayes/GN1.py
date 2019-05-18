# -*- coding: utf-8 -*-


###Task 手动构建贝叶斯的一些关键功能


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
# path = 'C:\\Users\\lmc\\Desktop\\测试-lmc\\练习题\\贝叶斯\\练习'
# os.chdir(path)

data = pd.read_csv('D:/py_codes/Mytask_machineLearning/task4_naive_bayes/vehicle.csv')

feature=data.iloc[:,:2].values
#print(feature)
feature_car = data[data['label']=='car'].iloc[:,:2].values
feature_truck = data[data['label']=='truck'].iloc[:,:2].values
# print(feature_car)
# print(feature_truck)
#以下返回值均为两个类别的对应，也就是说返回值应该为一个包含两个值的向量
#以下定义是按照计算层级定义的，你也可以自己定义自己的函数，直接顶一个包含很多函数的类去计算

feature_mean = np.mean(feature,axis=0)
feature_mean_car =  np.mean(feature_car,axis=0)
feature_mean_truck =  np.mean(feature_truck,axis=0)
# print(feature_mean)
# print(feature_mean_car)
# print(feature_mean_car)

feature_std=np.std(feature,ddof = 1,axis=0)
feature_std_car = np.std(feature_car,ddof = 1,axis=0)
feature_std_truck = np.std(feature_truck,ddof = 1,axis=0)
# print(feature_std)
# print(feature_std_car)
# print(feature_std_truck)


def gaussian_probability(x):   # 变量x为包含长、宽两个特征值的向量
	gaussian_pro1=(1/(np.sqrt(np.pi*2)*feature_std))
	gaussian_pro2=np.exp((-np.square(x-feature_mean)/2*np.square(feature_std)))
	gaussian_pro=gaussian_pro1*gaussian_pro2
	return gaussian_pro
x=np.array([5,2])
y=gaussian_probability(x)
# print('%.10f'% (y[0]))

p_car=len(feature_car)/len(feature)       # car先验概率
p_truck=len(feature_truck)/len(feature)    # truck先验概率
# print(p_car)
# print(p_truck)

class Pro():

	def __init__(self,p_x1x2_car,p_x1x2_truck,px1_x2):
		self.p_x1x2_car = p_x1x2_car
		self.p_x1x2_truck = p_x1x2_truck
		self.px1_x2 = px1_x2
	def p_label_x1x2(self):
		p_car_x1x2=(self.p_x1x2_car[0]*self.p_x1x2_car[1]*p_car)/(self.px1_x2[0]*self.px1_x2[1])
		p_truck_x1x2=(self.p_x1x2_truck[0]*self.p_x1x2_truck[1]*p_car)/(self.px1_x2[0]*self.px1_x2[1])
		return p_car_x1x2,p_truck_x1x2
p_x1x2_car = np.array([2,3])
p_x1x2_truck=np.array([4,5])
px1_x2=np.array([1,2])
pro=Pro(p_x1x2_car,p_x1x2_truck,px1_x2)
print(pro.p_label_x1x2())

def get_class(p_x1x2_car,p_x1x2_truck):
	if p_x1x2_car>=p_x1x2_truck:
		return 'car'
	else:
		return 'truck'


