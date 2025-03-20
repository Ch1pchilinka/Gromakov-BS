import numpy as np
import pandas as pd
import math
import torch

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, auc, roc_auc_score, log_loss, mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from rit.tree import DecisionTreeClassifier as MyDecisionTreeClassifier
from rit.tree import DecisionTreeRegressor as MyDecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rit.ensemble import RandomForestClassifier as MyRandomForestClassifier
from rit.ensemble import RandomForestRegressor as MyRandomForestRegressor

from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_classification, make_regression

from scipy.stats import f_oneway

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from graphviz import Digraph

from model import CustomBoost

def dependency_matrix_tree_ver_1(tree, n_features):
    matrix = np.zeros((n_features,  n_features))

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
            matrix[feature[node_id]][feature[children_left[node_id]]] += 1
            matrix[feature[node_id]][feature[children_right[node_id]]] += 1
        else:
            is_leaves[node_id] = True
    return matrix / (n_nodes - 1)

def dependency_matrix_tree_ver_2(tree, n_features):
    matrix = np.zeros((n_features,  n_features))

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
            matrix[feature[node_id]][feature[children_left[node_id]]] += 1 / np.exp(depth)
            matrix[feature[node_id]][feature[children_right[node_id]]] += 1 / np.exp(depth)
        else:
            is_leaves[node_id] = True
    return matrix / (n_nodes - 1)

def dependency_matrix_forest(model, n_features):
  matrix = np.zeros((n_features,  n_features))

  for tree in model.estimators_:
    matrix += dependency_matrix_tree_ver_1(tree, n_features)

  N = model.n_estimators
  matrix = matrix / N

  return matrix

def dependency_matrix_forest_2(model, n_features):
  matrix = np.zeros((n_features,  n_features))

  for tree in model.estimators_:
    matrix += dependency_matrix_tree_ver_2(tree, n_features)

  N = model.n_estimators
  matrix = matrix / N

  return matrix

def Visualise_classification_2D(X, y):
  cols = ['blue', 'red', 'green', 'yellow']
  plt.figure(figsize=(9,4))
  plt.xlim((np.min(X) - 0.5, np.max(X) + 0.5)),
  plt.ylim((np.min(X) - 0.5, np.max(X) + 0.5))

  for k in np.unique(y):
      plt.plot(X[y==k,0], X[y==k,1], 'o',
                label='класс {}'.format(k), color=cols[k])

  plt.legend(loc='best')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show()

def Visualise_regression_2D(X, y, xlim, ylim):
  plt.figure(figsize=[9, 4])
  plt.xlim((xlim[0], xlim[1]))
  plt.ylim((ylim[0], ylim[1]))

  sc = plt.scatter(X[:, 0], X[:, 1], c=y, s=5)

  plt.colorbar(sc)

  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show()

def func_reg(X, y, alpha):
  params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'n_estimators': [5000]}
  clf = GridSearchCV(MyRandomForestRegressor(random_state=0,
                                              n_estimators=params['n_estimators'],
                                              max_depth=params['max_depth'],
                                              max_leaf_nodes=2**max(params['max_depth']),
                                              bootstrap=True,
                                              oob_score=True,
                                              subforest_importance=True,
                                              normalize_importance=False,
                                              rit_alpha=alpha),
                   params,
                   scoring = lambda est, X, y: est.oob_score_, 
                   cv=[(np.arange(X.shape[0]), np.arange(0))])
  clf.fit(X, y)
  rf = clf.best_estimator_
  print("rit_alpha = {}, max_depth = {}, score = {}".format(rf.rit_alpha, rf.max_depth, rf.oob_score_))
  return rf

def func_class(X, y, alpha):
  params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15], 'n_estimators': [5000]}
  clf = GridSearchCV(MyRandomForestClassifier(random_state=0,
                                              n_estimators=params['n_estimators'],
                                              max_depth=params['max_depth'],
                                              max_leaf_nodes=2**max(params['max_depth']),
                                              bootstrap=True,
                                              oob_score=accuracy_score,
                                              subforest_importance=True,
                                              normalize_importance=False,
                                              rit_alpha=alpha),
                   params,
                   scoring = lambda est, X, y: est.oob_score_, 
                   cv=[(np.arange(X.shape[0]), np.arange(0))])
  clf.fit(X, y)
  rf = clf.best_estimator_
  print("rit_alpha = {}, max_depth = {}, score = {}".format(rf.rit_alpha, rf.max_depth, rf.oob_score_))
  return rf

def export_tree_to_dot(clf, feature_names=None):
    dot = Digraph()
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    # Генерируем цветовую палитру для фичей
    unique_features = np.unique(feature[feature >= 0])
    custom_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]
    feature_colors = {f: custom_colors[i % len(custom_colors)] 
                 for i, f in enumerate(unique_features)}

    # Добавляем легенду
    with dot.subgraph(name='legend') as legend:
        legend.attr(label='Legend', rank='sink')
        for f, color in feature_colors.items():
            feature_label = feature_names[f] if feature_names else f"Feature {f}"
            legend.node(f"legend_{f}", 
                        label=feature_label, 
                        shape="box", 
                        fillcolor=color, 
                        style="filled")

    # Создаем узлы
    for node_id in range(n_nodes):
        if children_left[node_id] == children_right[node_id]:
            # Листовой узел (серый цвет)
            class_label = np.argmax(values[node_id][0])
            label = f"Class {class_label}"
            dot.node(str(node_id), 
                    label=label, 
                    shape="box", 
                    style="filled", 
                    fillcolor="lightgray")
        else:
            # Разделяющий узел (цвет по фиче)
            f = feature[node_id]
            feature_label = feature_names[f] if feature_names else f"X[{f}]"
            label = f"{feature_label} <= {threshold[node_id]:.3f}"
            dot.node(str(node_id), 
                    label=label, 
                    shape="ellipse", 
                    style="filled", 
                    fillcolor=feature_colors[f])

    # Создаем связи
    for node_id in range(n_nodes):
        if children_left[node_id] != children_right[node_id]:
            dot.edge(str(node_id), str(children_left[node_id]), label="True")
            dot.edge(str(node_id), str(children_right[node_id]), label="False")

    return dot

def create_tree_view(tree, name):
    dot = export_tree_to_dot(tree, feature_names=["X1", "X2"])
    dot.render(name, view=True, format='png')

def separate_cb_reg(cb):
    trees = cb.estimators_
    trees_0 = [tree for tree in trees if (0 in tree.tree_.feature) and (1 not in tree.tree_.feature)]
    trees_1 = [tree for tree in trees if (1 in tree.tree_.feature) and (0 not in tree.tree_.feature)]

    model_0 = CustomBoost(n_estimators=len(trees_0), rit_alpha=cb.rit_alpha, max_depth=cb.max_depth, nu=cb.nu)
    model_0.estimators_ = trees_0
    model_1 = CustomBoost(n_estimators=len(trees_1), rit_alpha=cb.rit_alpha, max_depth=cb.max_depth, nu=cb.nu)
    model_1.estimators_ = trees_1
    return model_0, model_1

def hypothesis_check(rf, X_test, y_test):
    rf_0, rf_1 = separate_cb_reg(rf)

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['figure.figsize'] = (16.0, 16.0)
    plt.rcParams['font.size'] = 50

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)

    subs = []
    for sub in gs:
        subs.append(fig.add_subplot(sub))
    subs = np.array(subs).reshape(2, 2)

    x0 = X_test.transpose(1,0)[0]
    x1 = X_test.transpose(1,0)[1]
    y0_rf = rf_0.predict(X_test)
    y0_pr = x0
    y1_rf = rf_1.predict(X_test)
    y1_pr = x1

    subplotNames = ['Feature 0', 'Feature 1', 'Features', 'Features']

    for i, ax in enumerate(subs.flat):
        ax.annotate(
            text=subplotNames[i], 
            xy=(0.5, 1.05),
            xycoords='axes fraction',
            ha='center',
            fontsize=24
        )
        
        if i == 0:
            ax.set_xlabel('X0')
            ax.scatter(x0, y0_rf, 0.3, 'b', label='predicted')
            ax.scatter(x0, y0_pr, 0.3, 'r', label='hypothesis, y = x')
        if i == 1:
            ax.set_xlabel('X1')
            ax.scatter(x1, y1_rf, 0.3, 'r', label='predicted')
            ax.scatter(x1, y1_pr, 0.3, 'b', label='hypothesis, y = x')
        if i == 2:
            ax.set_xlabel('X1 * X2')
            ax.scatter(x0 * x1, y0_rf * y1_rf, 0.3, 'r', label='hypothesis, y = y1 * y2')
            ax.scatter(x0 * x1, x0 + x1, 0.3, 'b', label='y = x')
        if i == 3:
            ax.set_xlabel('X1 * X2')
            ax.scatter(x0 * x1, rf.predict(X_test), 0.3, 'r', label='predicted')
            ax.scatter(x0 * x1, x0 * x1, 0.3, 'b', label='y = x')
        
        # Настройка осей
        ax.set_ylabel('y')
        ax.legend()
    print("score = {}".format(r2_score(rf.predict(X_test), y_test)))
    print("Trees with feature 0: {}, trees with feature 1: {}, total trees: {}".format(len(rf_0.estimators_), len(rf_1.estimators_), len(rf.estimators_)))
    plt.tight_layout()
    plt.show()

def func_cb(X, y, alpha, task):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {'max_depth': [1,2,3,4,5,6,7,8,9], 'nu': [1, 10, 100, 200], 'max_leaf_nodes': [2**9], 'rit_alpha': [alpha]}
    cv = KFold(n_splits=2).split(X_train)
    clf = GridSearchCV(CustomBoost(base_estimator=None,
                                   n_estimators=1000,
                                   task=task),
                   params,
                   scoring = 'r2' if task else "accuracy", 
                   cv=cv,
                   verbose=0)
    clf.fit(X_train, Y_train)
    cb = clf.best_estimator_
    cb.fit(X_train, Y_train)
    print("rit_alpha = {}, max_depth = {}, nu = {}, score = {}".format(cb.rit_alpha, cb.max_depth, cb.nu, r2_score(Y_test, cb.predict(X_test)) if task else accuracy_score(Y_test, cb.predict(X_test))))
    return cb

