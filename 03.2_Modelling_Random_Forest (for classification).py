# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import seaborn as sns

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# +
import pandas as pd

X_train_processed = pd.read_csv("Loan_X_train_processed.csv")

y_train = pd.Series.from_csv('Loan_y_train.csv')

X_test_processed = pd.read_csv("Loan_X_test_processed.csv")
# -

X_train_processed = X_train_processed.drop('Unnamed: 0', axis=1)
X_test_processed = X_test_processed.drop('Unnamed: 0', axis=1)

# # Root_mean_squared_error function 

# +
from sklearn.metrics import mean_squared_error

def rmse_value(X, y, model):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# -

# # run the random forest classifier

# +
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
forest_clf.fit(X_train_processed, y_train)

train_rmse = rmse_value(X_train_processed, y_train, forest_clf)
print(train_rmse)
# -

# # let's do this with cross validation

# +
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    


# +
# %%capture

forest_scores = cross_val_score(forest_clf, X_train_processed, y_train,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

# -

display_scores(forest_rmse_scores)

# # use grid search to find optimum random forest variables
#

# +
from sklearn.model_selection import GridSearchCV

display_scores(forest_rmse_scores)
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train_processed, y_train)
# -

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

best_model = grid_search.best_estimator_

rmse_value(X_train_processed, y_train, best_model)

# # let's use a learning curve to see how this is going

# +
from sklearn.model_selection import train_test_split

def plot_learning_curves(X, y, model, title, starting_point):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(starting_point, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") 
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Training set size')
    plt.ylabel('RMSE')

    plt.show()


# -

plot_learning_curves(X_train_processed, y_train, best_model, "Best Random Forest predictor", 1)

# # let's use confusion matrix to evaluate the error  properly 

# +
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(best_model, X_train_processed, y_train, cv=3)

y_train_pred = y_train_pred.round()

confusion_mat = confusion_matrix(y_train, y_train_pred)

sns.set_style("white")
plt.matshow(confusion_mat, cmap=plt.cm.gray)
plt.show()

# +
row_sums = confusion_mat.sum(axis=1, keepdims=True)
normalised_confusion_mat = confusion_mat/row_sums

plt.matshow(normalised_confusion_mat, cmap=plt.cm.gray)
print(confusion_mat, "\n")
print(normalised_confusion_mat)
# -

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train_pred, y_train))

# # what is the precision, recall and f1?

# +
from sklearn.metrics import precision_score, recall_score, f1_score

print('the precision score is : ', precision_score(y_train_pred, y_train))
print('the recall score is : ', recall_score(y_train_pred, y_train))
print('the f1 score is : ', f1_score(y_train_pred, y_train))
# -

# # all in all, this isn't so bad! let's look at the precision recall curves

# +
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_pred, y_train)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
    plt.xlabel("Threshold")
    plt.legend(loc="lower left") 
    plt.ylim([0, 1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds) 
plt.show()
# -

# # now let's make prediction on X_test with best_model - this was the gridsearched forest classifier 

predictions = best_model.predict(X_test_processed)
predictions

predictions_df = pd.DataFrame(predictions)



