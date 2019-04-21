# -*- coding: utf-8 -*-

###Task 1 完成评估函数的定义
##输入变量为预测值及真实值，返回值相应评估指标，不允许使用除数据结构外的第三方库
#
# 真正(True Positive , TP)：被模型预测为正的正样本。--有病，查出来了
# 假正(False Positive , FP)：被模型预测为正的负样本。--没病，说你有病
# 假负(False Negative , FN)：被模型预测为负的正样本。--有病，没查出来（*)
# 真负(True Negative , TN)：被模型预测为负的负样本。--没病，检查正常
#
# 分类模型的评估:
# 准确率Accuracy = (TP+TN)/(TP+TP+FP+FN)
# 精确率Precision = (TP)/(TP+FP)
# 召回率Recall = (TP)/(TP+FN)
# F1-score = (2*Precision*Recall)/(Precision+Recall)

predicted_value = [1,0,1,1,0,1,0,1,0,1,1,1,1,1]  #预测值
true_value =      [1,1,0,1,1,1,0,1,0,1,1,1,1,1]  #真实值  1为正样本，0为负样本

def getTruePositive(predicted_value,true_value):
    tp = 0
    for i in range(len(predicted_value)):
        if true_value[i] == 1 and predicted_value[i] ==1: #被模型预测为正的正样本。
            tp += 1
    return tp

def getFalsePositive(predicted_value,true_value):
    fp = 0
    for i in range(len(predicted_value)):
        if true_value[i] == 0 and predicted_value[i] == 1: #被模型预测为正的负样本
            fp += 1
    return fp
def getFalseNegative(predicted_value,true_value):
    fn = 0
    for i in range(len(predicted_value)):
        if true_value[i] == 1 and predicted_value[i] == 0: #被模型预测为负的正样本。
            fn += 1
    return fn
def getTrueNegative(predicted_value,true_value):
    tn = 0
    for i in range(len(predicted_value)):
        if true_value[i] == 0 and predicted_value[i] == 0: #被模型预测为负的负样本
            tn += 1
    return tn


#precision
def metrics_precision(predicted_value,true_value):
    # 精确率Precision = (TP)/(TP+FP)
    tp = getTruePositive(predicted_value,true_value)
    fp = getFalsePositive(predicted_value,true_value)
    score = tp / (tp+fp)
    return score

#recall
def metrics_recall(predicted_value,true_value):
    # 召回率Recall = (TP) / (TP + FN)
    tp = getTruePositive(predicted_value,true_value)
    fn = getFalseNegative(predicted_value,true_value)
    score = tp/(tp+fn)
    return score
	
#f1_score
def distance_f1_score(predicted_value,true_value):
    # F1 - score = (2 * Precision * Recall) / (Precision + Recall)
    precision = metrics_precision(predicted_value, true_value)
    recall = metrics_recall(predicted_value,true_value)
    score = 2*precision*recall / (precision+recall)
    return score

#测试precision
print(metrics_precision(predicted_value,true_value))
#测试recall
print(metrics_recall(predicted_value,true_value))
#测试f1_score
print(distance_f1_score(predicted_value,true_value))
