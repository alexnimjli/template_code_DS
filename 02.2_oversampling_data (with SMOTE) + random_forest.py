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

X_train = pd.read_csv("Loan_train.csv")
# -

X_train.head()

X_train['Loan_Status'].unique()

X_train['Loan_Status'].value_counts()

# +
# change the varialbes Y and N into 1 and 0

X_train['Loan_Status'] = (X_train['Loan_Status'] == 'Y').astype(int)
# -

X_train['Loan_Status'].unique()

X_train['Loan_Status'].value_counts()

print('Gets loan:', round(X_train['Loan_Status'].value_counts()[1]/len(X_train) * 100,2), '% of the dataset')
print('Does not get loan:', round(X_train['Loan_Status'].value_counts()[0]/len(X_train) * 100,2), '% of the dataset')

# # before we use SMOTE to oversample the data, we need to fill in the missing values and preprocess the data for modelling

X_train.isnull().any(axis=0)

X_train.apply(lambda x: sum(x.isnull()),axis=0)


# # functions to fill in the nans in the columns with the mean or mode

# +
def fill_mode(df, attribute_list):
    for i in attribute_list:
        print(i)
        df[i].fillna(df[i].mode()[0], inplace=True)
    return df

def fill_mean(df, attribute_list):
    for i in attribute_list:
        print(i)
        df[i].fillna(df[i].mean(), inplace=True)
    return df


# -

X_train = fill_mode(X_train, ['Gender', 'Dependents', 'Self_Employed', "Married", 'Credit_History'])
X_train = fill_mean(X_train, ['LoanAmount', 'Loan_Amount_Term'])

X_train.apply(lambda x: sum(x.isnull()),axis=0)

X_train.dtypes

# # before we use SMOTE, we need to turn all categorical data points into numerical points (i.e. one hot encoding), while we're at it, let's use StandardScaler for the numerical features

# +
# separate the dataframe into X and y sets
y_train = X_train["Loan_Status"]
X_train = X_train.drop(["Loan_Status"], axis = 1)

# drop unecessary columns
X_train = X_train.drop(['Loan_ID'], axis = 1)

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class selector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
      
num_attributes = ["LoanAmount", "Loan_Amount_Term",
                  'ApplicantIncome', 'CoapplicantIncome']

num_pipeline = Pipeline([
            ('selector', selector(num_attributes)),
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
                    ])

cat_attributes = ['Gender', 'Married', 'Education', 'Self_Employed', 
                  'Property_Area', 'Dependents', 'Credit_History']

cat_pipeline = Pipeline([
                ('selector', selector(cat_attributes)),
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('cat_encoder', OneHotEncoder(sparse=False)),
])

# +
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
# -

X_train_processed = full_pipeline.fit_transform(X_train)
X_train_processed = pd.DataFrame(X_train_processed)
X_train_processed.head()

# # now that the data has been processed and all values are numerical, we can use SMOTE.
#
# # here is an example of SMOTE being used on data before cross-validation

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# +
print('Before OverSampling, the shape of X_train: {}'.format(X_train_processed.shape))
print('Before OverSampling, the shape of y_train: {} \n'.format(y_train.shape))

print("Before OverSampling, counts of label 'Y': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label 'N': {} \n".format(sum(y_train==0)))

smt = SMOTE()
X_train_processed_ovs, y_train_ovs = smt.fit_sample(X_train_processed, y_train)

print('After OverSampling, the shape of X_train: {}'.format(X_train_processed_ovs.shape))
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_ovs.shape))

print("After OverSampling, counts of label 'Y': {}".format(sum(y_train_ovs==1)))
print("After OverSampling, counts of label 'N': {}".format(sum(y_train_ovs==0)))
# -

# # think of doing stratified shuffle split! instead of KFold 

# from sklearn.model_selection import StratifiedShuffleSplit
#
#
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
# for train_index, test_index in split.split(X_train, y_train):
#     strat_train_set = X_train.loc[train_index] 
#     strat_test_set = X_train.loc[test_index]
#     2

# # but we should use SMOTE during cross validation! not before, as the synthetic data points will influence the test set and validation set and therefore skew our model
#
# # let's split the data into training and test data first

# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_processed, y_train, train_size= 0.6, random_state = 42, stratify=y_train)
# -

X_train.shape

y_train.value_counts()

X_test.shape

y_test.value_counts()

# # 'KFold + SMOTE + model' function 

# +
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

def rmse_value(X, y, model):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# -

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# # here is the KFold + SMOTE + model fitting function

# +
from sklearn.model_selection import KFold

def KFold_SMOTE_model_scores(X_df, y, model):
    
    scores = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    
    # need to reset the indices as the 
    X_df = X_df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    #this will shuffle through 10 different training and validation data splits 
    for train_index, val_index in cv.split(X_df):
        
        X_train = X_df.loc[train_index]
        y_train = y.loc[train_index]
        
        X_val = X_df.loc[val_index]
        y_val = y.loc[val_index]   
        
        print('Before OverSampling, the shape of X_train: {}'.format(X_train.shape))
        print('Before OverSampling, the shape of y_train: {} \n'.format(y_train.shape))

        print("Before OverSampling, counts of label 'Y': {}".format(sum(y_train==1)))
        print("Before OverSampling, counts of label 'N': {} \n".format(sum(y_train==0)))
        
        
        # this will create minority class data points such that y_train has 50% == 1 and 50% == 0
        X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)
        
        print('After OverSampling, the shape of X_train: {}'.format(X_train_SMOTE.shape))
        print('After OverSampling, the shape of y_train: {} \n'.format(y_train_SMOTE.shape))

        print("After OverSampling, counts of label 'Y': {}".format(sum(y_train_SMOTE==1)))
        print("After OverSampling, counts of label 'N': {} \n".format(sum(y_train_SMOTE==0)))
        
        print("---" * 7)
        print("\n")
        
        model.fit(X_train_SMOTE, y_train_SMOTE)
        
        #find the accuracy score of the validation set
        y_val_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_val_pred))
        
        #find the best model based on the accuracy score
        if accuracy_score(y_val, y_val_pred) == max(scores):
            best_model = model
    
    return scores, best_model

# +
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

scores, best_model = KFold_SMOTE_model_scores(X_train, y_train, forest_clf)
# -

scores = np.array(scores)
display_scores(scores)

# these scores are the accuracy of the classifier being tested on validation data

best_model

# # let's test this on test data that has not been processed by SMOTE

y_test.value_counts()



# +
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#y_train_pred = cross_val_predict(best_model, X_train_processed, y_train, cv=3)
y_test_pred = best_model.predict(X_test)
y_test_pred = y_test_pred.round()

confusion_mat = confusion_matrix(y_test, y_test_pred)

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
print(accuracy_score(y_test_pred, y_test))

# +
from sklearn.metrics import precision_score, recall_score, f1_score

print('the precision score is : ', precision_score(y_test_pred, y_test))
print('the recall score is : ', recall_score(y_test_pred, y_test))
print('the f1 score is : ', f1_score(y_test_pred, y_test))
# -
from sklearn.metrics import classification_report
print(classification_report(y_test_pred, y_test))


















# Let's look at the training curve for this

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

plot_learning_curves(X_train, y_train, best_model, "Best Random Forest predictor", 1)



# # make predictions

predictions = best_model.predict(X_test_processed)
predictions


