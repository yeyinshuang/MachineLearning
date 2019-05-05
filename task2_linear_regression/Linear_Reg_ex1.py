# -*- coding: utf-8 -*-


###Task 2 使用sklearn解决回归问题
##输入数据为linear_data.csv
#要求：
#（1）针对模型构建回归模型
#（2）可视化出回归直线
#（3）*使用你自己定义函数的计算出MSE，不允许使用sklearn的API
#（4）*** 选做，不计入考评分数，使用poly_data.csv，分别构建线性回归和多项式回归模型，计算RMSE，并进行可视化
# 提示：相关API：sklearn.preprocessing.PolynomialFeatures

########  你的可视化结果看起来应该类似于'demo/demo_poly'  ##############

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
# * from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('linear_data.csv')

labels = data['hx']
data = data['x'].values.reshape(-1,1)
# plt.scatter(data,labels,c='black')

from sklearn.linear_model import LinearRegression

model_plain = LinearRegression(normalize = True)
model_plain.fit(data,labels)

# 生成图像绘制数据
x_ploy = np.arange(0,11,0.01)
y_ploy = model_plain.predict(x_ploy.reshape(-1,1))
print(y_ploy)

plt.scatter(data,labels,c='black')
plt.plot(x_ploy,y_ploy,'-r')
plt.show()

from task2_linear_regression.metrics_ext import metrics_MSE

print(metrics_MSE(labels,model_plain.predict(data.reshape(-1,1))))




from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('poly_data.csv')

poly_features_3 = PolynomialFeatures(degree = 3)
model_poly = LinearRegression()
model_poly.fit(poly_features_3.fit_transform(data['x'].values.reshape(-1,1)), data['hx'].values.reshape(-1,1))

plot_x_poly = np.linspace(0, 8, 1000).reshape(-1, 1)
plot_y_poly = model_poly.predict(poly_features_3.fit_transform(plot_x_poly))

# plt.scatter(data['x'], data['hx'])
# plt.plot(plot_x_poly, plot_y_poly, 'r-')
# plt.xlabel("x")
# plt.ylabel("hx")

model_linear = LinearRegression()
model_linear.fit(data['x'].values.reshape(-1,1), data['hx'].values.reshape(-1,1))

plot_x_linear = np.linspace(0, 8, 1000).reshape(-1, 1)
plot_y_linear = model_linear.predict(plot_x_linear)

plt.scatter(data['x'], data['hx'])
plt.plot(plot_x_poly, plot_y_poly, 'r-')
plt.plot(plot_x_linear, plot_y_linear, 'b-')
plt.xlabel("x")
plt.ylabel("hx")
plt.legend(['poly','linear'])
plt.title('regression')
plt.show()














