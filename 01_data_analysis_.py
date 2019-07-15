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

np.random.seed(42)

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
pd.set_option('display.max_rows', 30)

IMDB_df = pd.read_csv("IMDB-Movie-Data.csv")
# -

IMDB_df.head()

# # function that plots scatter graph with regression line 

# +
from sklearn.linear_model import LinearRegression

def plot_scatter(df, x_attribute, y_attribute):
    df = df[np.isfinite(df[x_attribute])]
    df = df[np.isfinite(df[y_attribute])]

    x = df[x_attribute].values.tolist()
    x = np.reshape(x, (len(x),1))
    y = df[y_attribute].values.tolist()
    regr = LinearRegression()
    regr.fit(x, y)

    print("Regression coeff: {:0.2f}".format(regr.coef_[0]))

    fig,ax = plt.subplots()
    plt.xlim(min(x), max(x))
    plt.ylim(0, max(y))
    plt.title(x_attribute)
    ax.scatter(x, y, s=5, color = 'r')
    ax.plot(x, regr.predict(x), color='k')
    plt.xlabel(x_attribute)
    plt.ylabel(y_attribute)
    return


# -

plot_scatter(IMDB_df, 'Year', 'Rating')

plot_scatter(IMDB_df, 'Runtime (Minutes)', 'Rating')

# +
train_df = pd.read_csv("vehicle_train.csv")
test_df = pd.read_csv("vehicle_test.csv")

X_train = train_df
X_test = test_df
# -

X_train.head()


# # box plots to check for outliers

def box_plots(df, attribute_list):
    new_df = pd.DataFrame()
    for i in attribute_list:
        new_df[i] = df[i]
            
    new_df.boxplot(return_type='dict')
    plt.plot
    return 


# +
attribute_list = ['DISBURSED_AMOUNT', 'ASSET_COST', "LTV"]

box_plots(X_train, attribute_list)
# -

box_plots(X_train, ["LTV"])

X_train['LTV'].describe()

# +
attribute_list = ['Runtime (Minutes)']

box_plots(IMDB_df, attribute_list)


# -

def box_plot_outliers(df, attribute, max_val, min_val):
    attribute_column = df[attribute]
    outliers_max = (attribute_column > max_val)
    print(df[outliers_max])
    outliers_min = (attribute_column < min_val)
    print(df[outliers_min])
    return df[outliers_min]



box_plot_outliers(X_train, "LTV", 100, 45)

IMDB_df.columns

# # DBSCAN to remove outliers

from sklearn.cluster import DBSCAN
from collections import Counter

# +
df = IMDB_df
df = df.drop('Title', axis=1)
df = df.drop('Genre', axis=1)
df = df.drop('Description', axis=1)
df = df.drop('Director', axis=1)
df = df.drop('Actors', axis=1)

model = DBSCAN(eps=0.8, min_samples=19).fit(df)


# -

# # function that plots bar graphs 

def plot_bar_graphs(df, attribute):
    plt.figure(1)
    plt.subplot(131)
    df[attribute].value_counts(normalize=True).plot.bar(figsize=(22,4),title= attribute)
    
    crosstab = pd.crosstab(df[attribute], df['LOAN_DEFAULT'])
    crosstab.div(crosstab.sum(1).astype(float), axis=0).plot.bar(stacked=True)
    crosstab.plot.bar(stacked=True)
    return


plot_bar_graphs(X_train, "EMPLOYMENT_TYPE")

# # function that plots histograms with mean and medium - this one if you want to select n_bins and x_max

# +
dodger_blue = '#1E90FF'
crimson = '#DC143C'
lime_green = '#32CD32'
red_wine = '#722f37'
white_wine = '#dbdd46' 

def plot_histograms(df, x_attribute, n_bins, x_max):
    
    #this removes the rows with nan values for this attribute  
    df = df.dropna(subset=[x_attribute]) 
    
    print ("Mean: {:0.2f}".format(df[x_attribute].mean()))
    print ("Median: {:0.2f}".format(df[x_attribute].median()))
           
    df[x_attribute].hist(bins= n_bins, color= crimson)
    
    #this plots the mean and median 
    plt.plot([df[x_attribute].mean(), df[x_attribute].mean()], [0, 60000],
        color='black', linestyle='-', linewidth=2, label='mean')
    plt.plot([df[x_attribute].median(), df[x_attribute].median()], [0, 60000],
        color='black', linestyle='--', linewidth=2, label='median')
    
    plt.xlim(xmin=0, xmax = x_max)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.title(x_attribute)
    plt.legend(loc='best')
    plt.show()

    df[df['LOAN_DEFAULT']==0][x_attribute].hist(bins=n_bins, color = crimson, label='No default')

    print ("Y Mean: {:0.2f}".format(df[df['LOAN_DEFAULT']==0][x_attribute].mean()))
    print ("Y Median: {:0.2f}".format(df[df['LOAN_DEFAULT']==0][x_attribute].median()))
    
    plt.plot([df[df['LOAN_DEFAULT']==0][x_attribute].mean(), df[df['LOAN_DEFAULT']==0][x_attribute].mean()], 
            [0, 60000], color='r', linestyle='-', linewidth=2, label='Y mean') 
    plt.plot([df[df['LOAN_DEFAULT']==1][x_attribute].mean(), df[df['LOAN_DEFAULT']==1][x_attribute].mean()], 
            [0, 60000], color='b', linestyle='-', linewidth=2, label='N mean')
 
    df[df['LOAN_DEFAULT']==1][x_attribute].hist(bins=n_bins, color = lime_green, label='Default')
    
    print ("N Mean: {:0.2f}".format(df[df['LOAN_DEFAULT']==1][x_attribute].mean()))
    print ("N Median: {:0.2f}".format(df[df['LOAN_DEFAULT']==1][x_attribute].median()))
    
    plt.plot([df[df['LOAN_DEFAULT']==0][x_attribute].median(), df[df['LOAN_DEFAULT']==0][x_attribute].median()], 
            [0, 60000], color='r', linestyle='--', linewidth=2, label='Y median') 
    plt.plot([df[df['LOAN_DEFAULT']==1][x_attribute].median(), df[df['LOAN_DEFAULT']==1][x_attribute].median()], 
            [0, 60000], color='b', linestyle='--', linewidth=2, label='N median')
    
    plt.xlim(xmin=0, xmax = x_max)
    
    plt.title(x_attribute)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.legend(loc='best')
    plt.show()    
    return
    


# -

plot_histograms(X_train, 'DISBURSED_AMOUNT', 100, 150000)

# # function that plots histogram with mean and medium if there aren't many bins and sparsely populated data, so no need to specify n_bins and x_max

# +
dodger_blue = '#1E90FF'
crimson = '#DC143C'
lime_green = '#32CD32'
red_wine = '#722f37'
white_wine = '#dbdd46' 
    
def plot_histograms(df, attribute):
    
    #this removes the nan values for this attribute  
    df = df.dropna(subset=[attribute]) 
    
    print ("Mean: {:0.2f}".format(df[attribute].mean()))
    print ("Median: {:0.2f}".format(df[attribute].median()))
           
    df[attribute].hist(bins=len(df[attribute].unique()), color= crimson)
    
    #pd.value_counts(df[attribute]).max() counts the number maximum frequency 
    #for a unique value in the column attribute 
    
    plt.plot([df[attribute].mean(), df[attribute].mean()], [0, pd.value_counts(df[attribute], sort=True).max()],
        color='black', linestyle='-', linewidth=2, label='mean')
    plt.plot([df[attribute].median(), df[attribute].median()], [0, pd.value_counts(df[attribute], sort=True).max()],
        color='black', linestyle='--', linewidth=2, label='median')

    plt.xlim(xmin= 0, xmax = df[attribute].unique().max() +1)
    plt.xlabel(attribute)
    plt.ylabel('COUNT')
    plt.title(attribute)
    plt.legend(loc='best')
    plt.show()
    
    #df[df['LOAN_DEFAULT']==0][attribute].hist(bins=n_bins, color = crimson, label='No default')

    #df[df['LOAN_DEFAULT']==1][attribute].hist(bins=n_bins, color = lime_green, label='Default')
    
    #plt.xlim(xmin=0, xmax = df[attribute].unique().max())
    
    #plt.title(attribute)
    #plt.xlabel(attribute)
    #plt.ylabel('COUNT')
    #plt.legend(loc='best')
    #plt.show()    
    return
    


# -

plot_histograms(IMDB_df, 'Rating')

# # function for correlation matrix, here we can select which features to check correlation with

# +
import seaborn as sns

def corr_matrix(df, attribute_list, key_attribute):
    new_df = pd.DataFrame()
    for i in attribute_list:
        new_df[i] = df[i]
            
    matrix = new_df.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, vmax=.8, square=True, cmap="YlGnBu")
    
    print(matrix[key_attribute].sort_values(ascending=False))
    
    return 


# -

list_attribs = ['LOAN_DEFAULT', 'ASSET_COST', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'DISBURSED_AMOUNT']

corr_matrix(X_train, list_attribs, 'LOAN_DEFAULT')

IMDB_df.head()

list_attribs = ['Year', 'Runtime (Minutes)', 'Votes', 'Revenue (Millions)', 'Metascore', 'Rating']

corr_matrix(IMDB_df, list_attribs, 'Rating')


