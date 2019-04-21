# -*- coding: utf-8 -*-


###Task 2 使用sklearn解决分类问题
##输入数据为车辆数据vehicle
#要求：
#（1）分别使用k=3，5，9对目标数据集进行分类
#（2）可视化出分类结果，并标注被错误分类的点
#（3）呈现完整的可视化结果
#（4）计算f1值

########  你的可视化结果看起来应该类似于'answer_demo.png'  ##############

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data = pd.read_csv('vehicle.csv')
labels = data['label']
data = data.drop('label',axis=1)
# print(data)
from sklearn.model_selection import train_test_split
feature_train,feature_test,label_train,label_test = train_test_split(data,labels,test_size = 0.2,random_state=0) #分配训练和验证集


from sklearn.neighbors import KNeighborsClassifier #导入k临近分类
model = KNeighborsClassifier(n_neighbors=3)  #传入超参数k值 3,5,9
model.fit(feature_train,label_train)
prediction = model.predict(feature_test)
print(prediction)

from sklearn.metrics import classification_report
labels = ['car','truck']
classes = ['car','truck']
print(classification_report(label_test,prediction,target_names=classes,labels=labels,digits=4))#生成分类报告

import sklearn
# print(sklearn.metrics.accuracy_score(label_test,prediction))


print('f1: '+str(sklearn.metrics.f1_score(label_test,prediction,average='weighted'))) #weighted #macro #binary

import matplotlib.pyplot as plt

plt.plot()
plt.scatter(feature_test['length'][np.bitwise_and(label_test == prediction,label_test=='car')],
            feature_test['width'][np.bitwise_and(label_test== prediction,label_test=='car')],color='g')
plt.scatter(feature_test['length'][np.bitwise_and(label_test == prediction,label_test=='truck')],
            feature_test['width'][np.bitwise_and(label_test== prediction,label_test=='truck')],color='r')
plt.scatter(feature_test['length'][label_test!=prediction],feature_test['width'][label_test!=prediction],color='k')
plt.title('car_truck_classification_result')#显示图表标题
plt.xlabel('length')#x轴名称
plt.ylabel('width')#y轴名称
plt.legend(['car','truck','wrong'])
# plt.grid(True)#显示网格线
plt.show()



































