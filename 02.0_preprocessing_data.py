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


class DataProcessing:
    
    def __init__(self, data, target):
        self.data = data.replace({'':np.nan})
        #self.data = data
        self.target = target
        self.X = self.data.drop(target, axis = 1)
        self.y = self.data[target]
        
    def threshold_col_del(self, threshold):
        """
        This function keeps only columns that have a share of non-missing values above the threshold. 
        """
        self.data = self.data.dropna(thresh=threshold*len(self.data), axis=1) 
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
    
    def extract_timestamps(self, start_date = '2017-12-01'):
        """
        This function extracts different time stamps from the variable 'TransactionDT'
        such as day of the month, day of the week, hours and minutes and converts them
        into extra variables of the dataframe.
        """
        startdate = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.data["Date"] = self.data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        self.data['Day of the month'] = self.data['Date'].dt.day
        self.data['Day of the week'] = self.data['Date'].dt.dayofweek
        self.data['Hours'] = self.data['Date'].dt.hour
        self.data['Minutes'] = self.data['Date'].dt.minute
        self.data.drop('Date', axis = 1, inplace = True)
        
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
    
    def lblencoder(self):
        """
        This function replaces string variables with encoded values.
        """
        for i in self.data.columns:
            if self.data[i].dtype=='object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.data[i].values))
                self.data[i] = lbl.transform(list(self.data[i].values))
                
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
        
    def fill_null(self, attribute_list, stat, integer = -999): 
        """
        This function fills null values of selected columns with one of four different methods:
            - 'median' will fill the nulls with the median of the column. 
            - 'mean' uses the mean of the column. 
            - 'mode' uses the mode of the column. It can be used with string 
            variables, but they need to have been encoded first.
            - 'integer' fills the nulls with an integer (-999 by default).
        """
        for i in attribute_list:     
            if stat == 'median':
                self.data[i].fillna(self.data[i].median(), inplace=True) 
                self.data[i] = self.data[i].astype(float)
            elif stat == 'mean':
                self.data[i].fillna(self.data[i].mean(), inplace=True)
                self.data[i] = self.data[i].astype(float)
            elif stat == 'mode':
                self.data[i].fillna(self.data[i].mode()[0], inplace=True)     
                self.data[i] = self.data[i].astype(int)
            elif stat == 'integer':
                self.data[i].fillna(integer, inplace=True) 
                self.data[i] = self.data[i].astype(float)                
            #print(self.data[i].dtype)
        
        self.X = self.data.drop(self.target, axis =1)
        self.y = self.data[self.target]
    
    def balancesample(self, typ, rs=42):
        #Updating the self.X and self.y
        self.X = self.data.drop(self.target, axis = 1)
        self.y = self.data[self.target]
        # This conditional statement runs undersampling and oversampling
        # depending on the user's requirements.
        if typ == "under":
            rus = RandomUnderSampler(random_state=rs)
            self.X, self.y = rus.fit_resample(self.X, self.y)
        if typ == "over":
            ros = RandomOverSampler(random_state=rs)
            self.X, self.y = ros.fit_resample(self.X, self.y)

    def standardiser(self):
        """
        This function standardises the numeric columns of a dataframe. 
        """
        # Select only numeric features first

        #self.X = self.data.loc[:, self.data.columns != self.target].values
        numeric_columns = []
        for col in self.X.columns:
            if self.X[col].dtype!='object':
                numeric_columns.append(col)
        scaler = preprocessing.StandardScaler().fit(self.X[numeric_columns]) 
        # Now we can standardise
        self.X[numeric_columns] = scaler.transform(self.X[numeric_columns])       
                      
    def pca_reduction(self, variance):
        pca = PCA(n_components = variance)
        self.X = pca.fit_transform(self.X)
        self.X = pd.DataFrame(self.X)





X_train.isnull().sum()


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


