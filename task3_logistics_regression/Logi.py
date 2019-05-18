# -*- coding: utf-8 -*-


###Task 2 使用sklearn逻辑回归进行分类
##输入数据为vehicle.csv
#要求：
#（1）针对模型构建分类模型
#（2）画出分类边界
#提示：model.coef_ & model.intercept_


########  你的可视化结果看起来应该类似于'demo'  ##############

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('vehicle.csv')
labels = data['label']

# print(data[data['label'] == 'car'])
data = data.drop('label',axis=1)



# print(data)

# 分离数据集和测试机
from sklearn.model_selection import train_test_split
feature_train,feature_test,label_train,label_test = train_test_split(data,labels,test_size=0.2,random_state=3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
                            C=1.0,class_weight=None,multi_class='ovr',penalty='l2',solver='liblinear'
###solver 求解方法 默认为liblinear
## 一般情况下，对于小的数据集，liblinear较好；对于大的数据集，sag速度更快
#liblinear 坐标轴下降法来迭代优化损失函数
#lbfgs 利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
#newton-cg 牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
#sag 随机平均梯度下降

###C 正则化强度 默认为1.0

###penalty 正则化方法， L1正则与L2正则，
#L1， newton-cg, lbfgs，sag都要计算损失函数的导数，L1没有连续导数，所以...GG,只能用liblinear
#L2， newton-cg, lbfgs，sag，liblinear都可以使用

###multi_class 多分类方法 默认为ovr
#ovr OvR相对简单，但分类效果相对略差
#multinomial 效果相对会更好 但更耗时

###class_weight 分类权重 默认None
#接收列表类数据如[1,2]， 表示第一类权重为1，第二类权重为2，用于对付不平衡样本
)
model.fit(feature_train,label_train)
prediction = model.predict(feature_test)
print(prediction)

print(model.coef_)
print(model.intercept_)




from sklearn.metrics import f1_score
current_f1 = f1_score(prediction,label_test,average='macro')
print(current_f1)

plt.scatter(feature_test['length'][np.bitwise_and(label_test == prediction,label_test=='car')],
            feature_test['width'][np.bitwise_and(label_test== prediction,label_test=='car')],color='g')
plt.scatter(feature_test['length'][np.bitwise_and(label_test == prediction,label_test=='truck')],
            feature_test['width'][np.bitwise_and(label_test== prediction,label_test=='truck')],color='r')

plot_x_line = np.linspace(0,11,1000).reshape(-1,1)
plot_y_line = model.predict(plot_x_line)

plt.plot(plot_x_line,plot_y_line)

# plt.plot(la)
#决策边界没实现

plt.show()























