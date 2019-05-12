import pandas as pd
import numpy as np
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# data = data.drop('TARGET',axis=1)

# one-host encoding  把特征转换成多特征的数值
data = pd.get_dummies(data, dummy_na=True).fillna(0)  #np.mean(data)
data_test = pd.get_dummies(data_test, dummy_na=True).fillna(0)
labels = data['TARGET']
data = data.drop(['TARGET', 'ID'], axis=1)
data_test_labels = data_test[['ID', 'TARGET']]
data_test = data_test.drop(['TARGET', 'ID'], axis=1)
from sklearn.linear_model import LogisticRegression
logisticRegression_model = LogisticRegression()
logisticRegression_model.fit(data, labels)
prediction_test = logisticRegression_model.predict_proba(data_test)

data_test_labels['TARGET'] = prediction_test
print(data_test_labels)

pd.DataFrame(data_test_labels).to_csv('project_1.csv', index=False)


# labels = data[['TARGET','ID']]



# from sklearn.model_selection import train_test_split
# train_data, valid_train, train_labels, valid_labels = train_test_split(data, labels, test_size=0.2, random_state=1)


# print(train_labels)

# logisticRegression_model.fit(train_data, train_labels)
# prediction = logisticRegression_model.predict_proba(valid_train)[:, 1]
# from sklearn.metrics import roc_auc_score
# print(roc_auc_score(valid_labels, prediction))


# sub = pd.DataFrame()


# print(data_test_labels)




