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

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

classifier_output = pd.read_csv("classifier_output.csv")

# +
list_meta = []

full_len = 0
for i in range(40):
    a = str(i)
    if i < 10:
        meta = pd.read_csv("metadata/part-0000"+a+".csv")
    else:
        meta = pd.read_csv("metadata/part-000"+a+".csv")
    
    full_len = full_len + len(meta)
    list_meta.append(meta)    

for i in range(len(list_meta)):
    list_meta[i] = list_meta[i].merge(classifier_output, left_on = ['claim_id', 'part'], right_on = ['claim_id', 'part'], how = 'inner')


full_df = pd.concat(list_meta)

full_df.shape

# +
from matplotlib.pyplot import figure

X_train = full_df[full_df['set'] == 0]
y_train = X_train['operation']
part_train = X_train['part']
X_train = X_train.drop(columns = ['operation'])


cat_feats = [x for x in list(X_train.columns) if X_train[x].dtype == 'object']
num_feats = [x for x in list(X_train.columns) if x not in cat_feats]

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
    
num_pipeline = Pipeline([
            ('selector', selector(num_feats)),
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
                    ])

cat_pipeline = Pipeline([
                ('selector', selector(cat_feats)),
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('cat_encoder', OneHotEncoder(sparse=False)),
])

# +
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# +
X_train_processed = full_pipeline.fit_transform(X_train)
X_train_processed = pd.DataFrame(X_train_processed)

y_dummy = pd.get_dummies(y_train)['replace']


# +
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X_train_processed[:10000])

figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
colors = ['blue', 'red']
plt.scatter(X2D[:, 0], X2D[:, 1], c=y_dummy[:10000], zorder=10, s=2, cmap=matplotlib.colors.ListedColormap(colors), alpha = 0.4)
plt.colorbar()

# +
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
part_train_encoded = encoder.fit_transform(part_train)
part_train_encoded

encoder.classes_


# +
from sklearn.manifold import TSNE

figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

tsne = TSNE(n_components=2, random_state=0)
transformed_data = tsne.fit_transform(X_train_processed[:10000])
k = np.array(transformed_data)

#colors = ['blue', 'red']
plt.scatter(k[:, 0], k[:, 1], c=y_dummy[:10000], zorder=10, s=2, cmap=matplotlib.colors.ListedColormap(colors))

#for i, txt in enumerate(part_train[:10000]):
#    plt.annotate(txt, (k[i, 0], k[i, 1]))


# +
figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

plt.scatter(k[:, 0], k[:, 1], c=part_train_encoded[:10000], zorder=10, s=2)
plt.colorbar()
# -


