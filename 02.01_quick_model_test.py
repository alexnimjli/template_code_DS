# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#X_train = pd.read_csv("../01 - Data/unbalanced_X_train.csv")
#y_train = pd.read_csv('../01 - Data/unbalanced_y_train.csv', header = None, index_col = 0, squeeze = bool)
#X_train = X_train.drop('Unnamed: 0', axis=1)
#y_train = pd.read_csv("../01 - Data/unbalanced_y_train.csv", header = None)

# +
df_train = pd.read_csv('../01 - Data/df_train_split_ppc.csv')
df_test = pd.read_csv('../01 - Data/df_test_split_ppc.csv')

y_train = df_train['isFraud']
X_train = df_train.drop('isFraud', axis = 1)
y_test = df_test['isFraud']
X_test = df_test.drop('isFraud', axis = 1)
# -

import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.metrics import confusion_matrix
import seaborn as sns
def confusion_matrices(y, y_pred):
    y_pred = y_pred.round()
    confusion_mat = confusion_matrix(y, y_pred)
    sns.set_style("white")
    plt.matshow(confusion_mat, cmap=plt.cm.gray)
    plt.show()
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    normalised_confusion_mat = confusion_mat/row_sums
    print(confusion_mat, "\n")
    print(normalised_confusion_mat)
    plt.matshow(normalised_confusion_mat, cmap=plt.cm.gray)
    plt.show()
    print('the precision score is : ', precision_score(y, y_pred))
    print('the recall score is : ', recall_score(y, y_pred))
    print('the f1 score is : ', f1_score(y, y_pred))
    print('the accuracy score is : ', accuracy_score(y, y_pred))
    return


# ### for classification

# +
GaussianNB = GaussianNB()
SGDClassifier = SGDClassifier()
RandomForest = RandomForestClassifier(n_estimators=10)
XGBClassifier = XGBClassifier() 

scores_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fit_time'])

models = [GaussianNB, SGDClassifier, RandomForest, XGBClassifier]
names = ["Naive Bayes", "SGD Classifier", 'Random Forest Classifier', 'XGBClassifier']

for model, name in zip(models, names):
    temp_list = []
    print(name)

    model.fit(X_train, y_train)    
    scores = cross_validate(model, X_train, y_train,
                            scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'),
                            return_train_score=True, cv=10)
    
    for score in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        mean_score = scores['test_'+score].mean()  
        print('{} mean : {}'.format(score, mean_score))
        temp_list.append(mean_score)
    
    temp_list.append(scores['fit_time'].mean())
    print('average fit time: {}'.format(scores['fit_time'].mean()))
    print("\n")
    scores_df.loc[name] = temp_list
    
# -

# ### for regression

# +
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

lin_reg = LinearRegression()
dec_tree_reg = DecisionTreeRegressor()
simple_vec_reg = SVR(kernel='rbf', gamma='auto')
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

scores_df = pd.DataFrame(columns = ['average_MAE', 'average_RMSE', 'test_RMSE', 'fit_time'])

models = [lin_reg, dec_tree_reg, simple_vec_reg, forest_reg]
names = ["Linear Regression", "Decision Tree Regressor", "Simple Vector Regressor",'Random Forest Regressor']

for model, name in zip(models, names):
    temp_list = []
    print(name)
    
    t0 = time.time()
    model.fit(X_train, y_train)
    scores = cross_validate(model, X_train, y_train,
                            scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error'), cv=5)

    for score in ['neg_mean_absolute_error', 'neg_root_mean_squared_error']:
        mean_score = -scores['test_'+score].mean()
        print('CV {} mean : {}'.format(score, mean_score))
        temp_list.append(mean_score)

    test_predict = model.predict(tv_test_features)
    test_MSE = mean_squared_error(test_scores, test_predict)
    test_RMSE = np.sqrt(test_MSE)
    temp_list.append(test_RMSE)
    print('test RMSE: {}'.format(test_RMSE))
    
    t1 = time.time()
    temp_list.append(t1-t0)
    print('runtime: {} \n'.format(t1-t0))
    scores_df.loc[name] = temp_list

# -

scores_df

for i in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fit_time']:
    scores_df.plot.bar(y = i) 

# +
test_scores = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
    
for model, name in zip(models, names):
    temp_list = []    
    y_test_pred = model.predict(X_test)
    
    temp_list.append(accuracy_score(y_test, y_test_pred))
    temp_list.append(precision_score(y_test, y_test_pred))
    temp_list.append(recall_score(y_test, y_test_pred))
    temp_list.append(f1_score(y_test, y_test_pred))
    temp_list.append(roc_auc_score(y_test, y_test_pred))
    
    test_scores.loc[name] = temp_list

test_scores
# -

y_test_pred = GaussianNB.predict(X_test)
print('Naive Bayes')
confusion_matrices(y_test, y_test_pred)

y_test_pred = SGDClassifier.predict(X_test)
print('SGDClassifier')
confusion_matrices(y_test, y_test_pred)

y_test_pred = XGBClassifier.predict(X_test)
print('XGBClassifier')
confusion_matrices(y_test, y_test_pred)

y_test_pred = RandomForest.predict(X_test)
print('RandomForest')
confusion_matrices(y_test, y_test_pred)





# # doing this with cross_val_predict

# scores_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'run_time'])
#
# models = [GaussianNB(), RandomForestClassifier(n_estimators=10), 
#             SGDClassifier(), XGBClassifier()]
# names = ["Naive Bayes", "Random Forest Classifier", "SGD Classifier", "XGBClassifier"]
#
# #models = [GaussianNB(), SGDClassifier()]
# #names = ['Naive Bayes', 'SGD Classifier']
#
# for model, name in zip(models, names):
#     temp_list = []
#     print(name)
#     start = time.time()
#     
#     y_train_pred = cross_val_predict(model, X_train, y_train, cv=10)
#     
#     for score in ["accuracy", "precision", "recall", "f1", 'roc_auc']:
#         if score == 'accuracy':
#             mean_score = accuracy_score(y_train, y_train_pred)
#         elif score == 'precision':
#             mean_score = precision_score(y_train, y_train_pred)          
#         elif score == 'recall':
#             mean_score = recall_score(y_train, y_train_pred)
#         elif score == 'f1':
#             mean_score = f1_score(y_train, y_train_pred)
#         elif score == 'auc':
#             mean_score = auc_roc_score(y_train, y_train_pred)
#         
#         print('{} mean : {}'.format(score, mean_score))
#         temp_list.append(mean_score)
#     
#     #doing it this way takes more time, because it runs cross_val_score 4 different times
#     #for score in ["accuracy", "precision", "recall", "f1"]:
#      #   mean_score = cross_val_score(model, X_train, y_train, scoring=score, cv=10).mean()
#      #   print('{} mean score: {}'.format(score, mean_score))
#     
#     
#     temp_list.append(time.time() - start)
#     print("time to run: {}".format(time.time() - start))
#     print("\n")
#     scores_df.loc[name] = temp_list
