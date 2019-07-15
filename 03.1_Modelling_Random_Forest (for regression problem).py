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

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# +
import pandas as pd

X_train_processed = pd.read_csv("IMDB_X_train_processed.csv")
X_test_processed = pd.read_csv("IMDB_X_test_processed.csv")
y_train = pd.Series.from_csv("IMDB_y_train.csv")
y_test = pd.Series.from_csv("IMDB_y_test.csv")

# +
from sklearn.metrics import mean_squared_error

def rmse_value(X, y, model):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# -

# # Let's model the data with the random forest

# +
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(X_train_processed, y_train)

train_rmse = rmse_value(X_train_processed, y_train, forest_reg)
print(train_rmse)
# -

# use cross validation to get a better grasp on with the score

# +
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation of the scores:", scores.std())
    
forest_scores = cross_val_score(forest_reg, X_train_processed, y_train,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
# -

# this is pretty good, average RMSE of 0.68

# #  let's try to use grid search for random forest, we may find a better model! 

# +
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train_processed, y_train)
# -

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

best_model = grid_search.best_estimator_

# so random forest with max_features = 8 and n_estimators = 30 gives us the best result

# # let's build a learning curve just to see how this best random forest model trains itself, note the learning curve splits the training data into new training and val data

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
    plt.xlabel("Number of training samples")
    plt.ylabel("Mean squared error")
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") 
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc='best')
    plt.title(title)


# -

plot_learning_curves(X_train_processed, y_train, best_model, "Best Random Forest predictor", 1)

# # not bad, this is the best we can do with the data we have. not a high bias problem. 
#
# # anyway, let's make some predictions

# +
some_data = X_test_processed[20:30]

print("Predictions:", best_model.predict(some_data))
# -

print(y_test[20:30])

# looks like the scores are about -1 to +1 off, let's check this visually with a graph 

# +
x = []
for i in range(200):
    x.append(i) 
    
predictions = best_model.predict(X_test_processed)
y_test_array = y_test.as_matrix()

y = []
for i in range(200):
    y.append(predictions[i]-y_test_array[i])
    
y.sort()
plt.plot(x,y)
plt.ylabel("difference in rating")
plt.legend(loc='best')
plt.title("Difference between predicted and real value of y_test in ascending order ")
# -

# so the error in rating is roughly -1 to +1 with some outliers 








