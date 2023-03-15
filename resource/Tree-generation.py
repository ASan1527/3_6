# Random Forest Classifier
# Importing the libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data_deal import data_deal
from sklearn.metrics import classification_report
from rule_extraction import rule_extract
from rule_extraction import _tree_to_rules
from sklearn import tree
# Importing the datasets

'''
本文件用于测试集成决策树的能力
'''

datasets = pd.read_csv('../data/train_loan_prediction.csv')

# Deal with dataset
datasets = data_deal(datasets)



target_column = 'Loan_Status'
feature_columns = [column for column in list(datasets) if column != target_column]
Y = datasets[target_column]
X = datasets[feature_columns]

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Feature Scaling
# sc_X = StandardScaler()
# X_Train = sc_X.fit_transform(X_Train)
# X_Test = sc_X.transform(X_Test)


# Fitting the classifier into the Training set
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state =0, max_depth= 5)
classifier = classifier.fit(X_Train,Y_Train)


# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Create a confusion matrix
print(classification_report(Y_Test, Y_Pred))


clf = classifier.estimators_

rule_dict = _tree_to_rules(clf)
print(rule_dict.rule)

# print(rule_dict.rule_p)

