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

