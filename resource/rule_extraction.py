import os
import sklearn
from sklearn.tree import _tree
from warnings import warn
from sklearn import tree
import pydotplus
from six import StringIO
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


def rule_extract(model, feature_names, x_test=None, y_test=None, sort_key=0, recall_min_c1=0., precision_min_c1=0.,
                 recall_min_c0=0., precision_min_c0=0.):
    """
    Extract rules from Tree based Algorithm and filter rules based
    on recall/precision threshold on a given dataset.
    Return rules and rules with its performance.
    Parameters
    ----------
    model :
        用户训练好的模型
    feature_names:
        特征名称
    x_test : pandas.DataFrame.
        用来测试的样本的特征集
    y_test : pandas.DataFrame.
        用来测试的样本的y标签
    sort_key: 按哪一个指标排序，default = 0
            0： 按label=1样本的 recall 降序
            1： 按label=1样本的 precision 降序
            2： 按label=0样本的 recall 降序
            3： 按label=0样本的 precision 降序
    recall_min_c1：对1类样本的召回率表现筛选
    precision_min_c1：对1类样本的准确率表现筛选
    recall_min_c0：对0类样本的召回率表现筛选
    precision_min_c0：对0类样本的准确率表现筛选
    Returns
    -------
    rules: list of str.
        eg: ['Fare > 26 and Age > 10','Fare <= 20 and Age <= 18']
    rules_dict: list of tuples (rule, recall on 1-class, prec on 1-class, recall on 0-class, prec on 0-class, nb).
        eg:[('Fare > 26 and Age > 10', (0.32, 0.91, 0.97, 0.6, 1)),
            ('Fare <= 20 and Age <= 18', (0.22, 0.71, 0.17, 0.5, 1))]
    """

    rules_dict = {}
    rules_ = []
    # feature_names = feature_names if feature_names is not None
    #                      else ['X' + x for x in np.arange(X.shape[1]).astype(str)]

    # if input model is a single decision tree/classifier
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        rules_from_tree = _tree_to_rules(model, feature_names)

        # if no test data is given, return the rules
        if x_test is None:
            rules_ = [(r) for r in set(rules_from_tree)]
            return rules_, rules_dict
        # if test data is given, filter the rules based on precision/recall
        else:
            rules_from_tree = [(r, _eval_rule_perf(r, x_test, y_test)) for r in set(rules_from_tree)]
            rules_ = rules_from_tree

            # 根据评估指标进行规则筛选
            rules_dict = _rule_filter(rules_, rules_dict, recall_min_c1, precision_min_c1, recall_min_c0,
                                      precision_min_c0)

            # 按 recall_1 降序排列
            rules_dict = sorted(rules_dict.items(),
                                key=lambda x: (x[1][sort_key]), reverse=True)
            rules_ = [i[0] for i in rules_dict]
            return rules_, rules_dict

    elif isinstance(model, (sklearn.ensemble.bagging.BaggingClassifier,
                            sklearn.ensemble.bagging.BaggingRegressor,
                            sklearn.ensemble.forest.RandomForestClassifier,
                            sklearn.ensemble.forest.RandomForestRegressor,
                            sklearn.ensemble.forest.ExtraTreesClassifier,
                            sklearn.ensemble.forest.ExtraTreeRegressor)):
        if x_test is None:
            for estimator in model.estimators_:
                rules_from_tree = _tree_to_rules(estimator, feature_names)
                rules_from_tree = [(r) for r in set(rules_from_tree)]
                rules_ += rules_from_tree
            return rules_, rules_dict
        else:
            for estimator in model.estimators_:
                rules_from_tree = _tree_to_rules(estimator, feature_names)
                rules_from_tree = [(r, _eval_rule_perf(r, x_test, y_test)) for r in set(rules_from_tree)]
                rules_ += rules_from_tree

            # 根据评估指标进行规则筛选
            rules_dict = _rule_filter(rules_, rules_dict, recall_min_c1, precision_min_c1, recall_min_c0,
                                      precision_min_c0)

            # 按 recall_1 降序排列
            rules_dict = sorted(rules_dict.items(),
                                key=lambda x: (x[1][sort_key]), reverse=True)
            rules_ = [i[0] for i in rules_dict]
            return rules_, rules_dict

    else:
        raise ValueError('Unsupported model type!')
        return


# 2018.10.15 对规则筛选。目前由于测试集固定，实际上同一条规则的表现不会波动。
def _rule_filter(rules_, rules_dict, recall_min_c1, precision_min_c1, recall_min_c0, precision_min_c0):


    return rules_dict





def _tree_to_rules(tree):
    """
    Return a list of rules from a tree
    Parameters
    ----------
        tree : Decision Tree Classifier/Regressor
        feature_names: list of variable names
    Returns
    -------
    rules : list of rules.
    """
    class Rule_Dict:
        def __init__(self, rule, rule_p):
            self.rule = rule
            self.rule_p = rule_p
    rule = []
    rule_p = []
    R_D = Rule_Dict(rule, rule_p)
    n = 0
    def recurse(node, rule_element):


        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            element = [0,0,0]
            name = tree_.feature[node]
            # symbol = '<='
            # symbol2 = '>'
            threshold = tree_.threshold[node]
            if tree_.feature[node] in [0, 1, 2, 3, 5, 6]:
                threshold = int(threshold)
            else:
                threshold = round(threshold, 2)

            # 为该结点创造一个逻辑的列表(左孩子结点）
            element[0] = name
            element[1] = 0
            element[2] = threshold
            rule_element.append(element)
            # rule_element1 = rule_element.copy()
            print('rule-l', rule_element)
            rule_element1 = rule_element.copy()
            recurse(tree_.children_left[node], rule_element1)

            # 确定是否存在右孩子结点，避免重复输出规则
            if tree_.children_right[node] != -1:
                rule_element1[-1][1] = 1
                print('rule-r', rule_element)
                rule_element1 = rule_element.copy()
                recurse(tree_.children_right[node], rule_element1)
        else:
            rule.append(rule_element.copy())
            print('out', rule)
            if tree_.value[node][0][0] > tree_.value[node][0][1]:
                value = 0
            elif tree_.value[node][0][0] < tree_.value[node][0][1]:
                value = 1
            else:
                value = 'useless'
            R_D.rule_p.append(value)

    for n in range(2):
        # ie max_features != None
        tree_ = tree[n].tree_
        recurse(0, [])
    return R_D

# def rule_output(node, base_name):
#         rule = []
#         rule_element = []
#
#             if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             element = np.zeros([3])
#             name = feature_name[node]
#             symbol = '<='
#             symbol2 = '>'
#             threshold = tree_.threshold[node]
#
#                 # if name == 'Fare':
#                 #     threshold = round(threshold, 2)
#
#                 # 为该结点创造一个逻辑的列表(左孩子结点）
#                 element[0] = tree_.feature[node]
#                 element[1] = 0
#                 element[2] = threshold
#                 rule_element.append(element)
#                 text = base_name + ["{} {} {}".format(name, symbol, threshold)]
#                 recurse(tree_.children_left[node], text)
#
#                 # 确定是否存在右孩子结点，避免重复输出规则
#                 if tree_.children_right[node] != -1:
#                     rule_element[-1][1] = 1
#                     text = base_name + ["{} {} {}".format(name, symbol2, threshold)]
#                     recurse(tree_.children_right[node], text)
#                 rule_element.pop()
#             else:
#                 rule.append(rule_element)
#                 if tree_.value[node][0][0] > tree_.value[node][0][1]:
#                     value = 0
#                 elif tree_.value[node][0][0] < tree_.value[node][0][1]:
#                     value = 1
#                 else:
#                     value = 'useless'
#                 rule_dict[rule] = value
#
#                 yes = tree_.value[node][0][0]
#                 no = tree_.value[node][0][1]
#                 total = yes + no
#                 gini: float = 1 - (yes / total) ** 2 - (no / total) ** 2
#                 rule_txt = str.join(' and ', base_name)
#                 rule_txt = (rule_txt if rule_txt != ''
#                         else ' == '.join([feature_names[0]] * 2))
#                 rule_txt = 'IF  '"" + rule_txt + '  THEN  ' + 'Class=' + str(value) + '  Gini=' + str(gini)
#                 # a rule selecting all is set to "c0==c0"
#                 rules_txt.append(rule_txt)
#             # return rules_txt if len(rules_txt) > 0 else 'True'
#             return rule_dict
#
#
#         recurse(0, [] ,rule_dict)
#     return rule_dict
