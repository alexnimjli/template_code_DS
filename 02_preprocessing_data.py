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

train_df = pd.read_csv("Loan_train.csv")
test_df = pd.read_csv("Loan_test.csv")
# -

X_train = train_df
X_test = test_df

X_train.isnull().any(axis=0)

X_train.apply(lambda x: sum(x.isnull()),axis=0)


# # these functions will fill the nan values with the mean or mode of the column 

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

X_train['Credit_History'].unique()

X_train['Credit_History'].value_counts().to_dict()

fill_mode(X_train, ['Gender', 'Dependents', 'Self_Employed', "Married", 'Credit_History'])
fill_mean(X_train, ['LoanAmount', 'Loan_Amount_Term'])

X_train.apply(lambda x: sum(x.isnull()),axis=0)

fill_mode(X_test, ['Gender', 'Dependents', 'Self_Employed', "Married", 'Credit_History'])
fill_mean(X_test, ['LoanAmount', 'Loan_Amount_Term'])

X_train['Loan_Status'].unique()

X_train['Loan_Status'] = (X_train['Loan_Status'] == 'Y').astype(int)

X_train['Loan_Status'].unique()

# # drop unnecessary columns 

X_train = X_train.drop('Loan_ID', axis=1)
X_test = X_test.drop('Loan_ID', axis=1)

# # making new columns from original columns

# +
X_train['Total_Income']=X_train['ApplicantIncome']+X_train['CoapplicantIncome']
X_test['Total_Income']=X_test['ApplicantIncome']+X_test['CoapplicantIncome']

X_train['EMI']=X_train['LoanAmount']/X_train['Loan_Amount_Term']
X_test['EMI']=X_test['LoanAmount']/X_test['Loan_Amount_Term']
# -


