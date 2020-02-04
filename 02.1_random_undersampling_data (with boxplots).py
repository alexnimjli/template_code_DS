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

train_df = pd.read_csv("Loan_train.csv")
test_df = pd.read_csv("Loan_test.csv")
# -

X_train = train_df
X_test = test_df

X_train['Loan_Status'].value_counts()

print('Gets loan', round(X_train['Loan_Status'].value_counts()["Y"]/len(X_train) * 100,2), '% of the dataset')
print('Does not get loan', round(X_train['Loan_Status'].value_counts()['N']/len(X_train) * 100,2), '% of the dataset')

# # we want 50% 50% to make this data set less skewed, however, this comes at the risk of throwing away 230 'Y' values. That is ~37% of our dataset
#
# # first let's fill in the nan values 

X_train.head()


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

# # Let's have a look at the box plots of the values that could have outliers that we don't want 

# +
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=X_train, ax=axes[0])
axes[0].set_title('ApplicantIncome')

sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=X_train, ax=axes[1])
axes[1].set_title('CoapplicantIncome')

sns.boxplot(x="Loan_Status", y="LoanAmount", data=X_train, ax=axes[2])
axes[2].set_title('LoanAmount')

sns.boxplot(x="Loan_Status", y="Loan_Amount_Term", data=X_train, ax=axes[3])
axes[3].set_title('Loan_Amount_Term')

plt.show()


# -

# # function to remove outliers 

def remove_outliers(df, x_attrib, y_attrib):

    for i in X_train[y_attrib].unique():
        
        m, n = df.shape
        print('Number of rows: {}'.format(m))
        
        remove_list = df[x_attrib].loc[df[y_attrib] == i].values
        q25, q75 = np.percentile(remove_list, 25), np.percentile(remove_list, 75)
        print('Lower Quartile: {} | Upper Quartile: {}'.format(q25, q75))
        iqr = q75 - q25
        print('iqr: {}'.format(iqr))

        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        print('Cut Off: {}'.format(cut_off))
        print('Lower Extreme: {}'.format(lower))
        print('Upper Extreme: {}'.format(upper))

        outliers = [x for x in remove_list if x < lower or x > upper]
        print('Number of Outliers for {} Cases: {}'.format(i, len(outliers)))
        print('outliers:{}'.format(outliers))

        for d in outliers:
            #delete_row = new_df[new_df[y_attrib]==i].index
            #new_df = new_df.drop(delete_row)
            df = df[df[x_attrib] != d]
        
        m, n = df.shape
        print('Number of rows for new dataframe: {}\n'.format(m))
    
    new_df = df
    
    print('----' * 27)
    return new_df


# # simple box plot function that also prints out the medians

# +
def simple_box_plot(df, x_attrib, y_attrib):
    f, axes = plt.subplots(ncols=1, figsize=(7,7))

    sns.boxplot(x=y_attrib, y=x_attrib, data=df)
    axes.set_title(x_attrib)

    
    for i in X_train[y_attrib].unique():
        print("Median for '{}'': {}".format(i, df[x_attrib][df[y_attrib] == i].median()))

plt.show()
    
# -

# # before we remove outliers, let's look at the box plot in more detail before and after
#
# # First with ApplicantIncome

simple_box_plot(X_train, 'ApplicantIncome', 'Loan_Status')

X_train = remove_outliers(X_train, "ApplicantIncome", "Loan_Status")

simple_box_plot(X_train, 'ApplicantIncome', 'Loan_Status')

# # Now with CoapplicantIncome

simple_box_plot(X_train, 'CoapplicantIncome', 'Loan_Status')

X_train = remove_outliers(X_train, "CoapplicantIncome", "Loan_Status")

simple_box_plot(X_train, 'CoapplicantIncome', 'Loan_Status')

# # Now with LoanAmount

simple_box_plot(X_train, 'LoanAmount', 'Loan_Status')

X_train = remove_outliers(X_train, "LoanAmount", "Loan_Status")

simple_box_plot(X_train, 'LoanAmount', 'Loan_Status')

# # okay, removed all the outliers for the important variables - leave the Loan_Amount_Term
#
# # Let's check the value_counts to see the skew

X_train['Loan_Status'].value_counts()

print('Gets loan', round(X_train['Loan_Status'].value_counts()["Y"]/len(X_train) * 100,2), '% of the dataset')
print('Does not get loan', round(X_train['Loan_Status'].value_counts()['N']/len(X_train) * 100,2), '% of the dataset')

# # Of course, it's still skewed, but now we've removed all the outliers, we can make a dataframe that is 50/50

# +
n = X_train['Loan_Status'].value_counts()[1]

X_train = X_train.sample(frac=1)
fraud_df = X_train[X_train['Loan_Status'] == "N"]
non_fraud_df = X_train[X_train['Loan_Status'] == "Y"][:n]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
X_train = normal_distributed_df.sample(frac=1, random_state=42)

X_train.head()
# -

X_train['Loan_Status'].value_counts()

print('Gets loan', round(X_train['Loan_Status'].value_counts()["Y"]/len(X_train) * 100,2), '% of the dataset')
print('Does not get loan', round(X_train['Loan_Status'].value_counts()['N']/len(X_train) * 100,2), '% of the dataset')


