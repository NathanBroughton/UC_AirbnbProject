# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:10:10 2022
@author: NatBr
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn import svm, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.ticker import ScalarFormatter
import os
import time
import matplotlib.pyplot as plt
from collections import Counter
import sys
import seaborn as sns
import json
import math
from tqdm import tqdm
from esda.moran import Moran
from libpysal.weights import full2W

pd.options.mode.chained_assignment = None

ROS = RandomOverSampler()
RUS = RandomUnderSampler()
SM = SMOTE()

#Initiate training and test set
train_set = []
test_set = []
for file in os.listdir('./Data3/Papertrain'):
        path = os.path.join('./Data3/Papertrain',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        train_set.append(df)
for file in os.listdir('./Data3/Test'):
        path = os.path.join('./Data3/Test',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        test_set.append(df)
        # If no space in city name
        city = str(file[:-4])
        # If space in city name
        # city = "New York"
        # print(city)

df_train = pd.concat(train_set,axis=0,ignore_index=True)
df_test = pd.concat(test_set,axis=0,ignore_index=True)

df_all = pd.concat([df_train,df_test],axis=0,ignore_index=True)
train_shape = df_train.shape

tracts = list(df_test["NAME"])

#Plot penetration count histogram
Y = df_all['Airbnb penetration']
w = Counter(Y)

#Convert penetration number to class
df_all.to_csv('df_all.csv')
smallest = df_all.nsmallest(int(0.7*len(df_all)),['Airbnb penetration']).index
largest = df_all.nlargest(int(0.05*len(df_all)),['Airbnb penetration']).index

df_all['Airbnb penetration_class'] = [1]*len(df_all)
for i in smallest:
    df_all['Airbnb penetration_class'][i] = 0

for i in largest:
    df_all['Airbnb penetration_class'][i] = 2

#Start explanotary analysis
X = df_test[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center','Airbnb penetration']]

y = df_all['Airbnb penetration_class']

print("Skewness before")
print(X.skew(axis=0))

# Adjust skewness (comment the features which do not have to be transformed)
# X['Population Density'] = X['Population Density']**(1/2)
# X['Race Diversity Index'] = X['Race Diversity Index']**(1/2)
# X['Income Diversity Index'] = X['Income Diversity Index']**(1/2)
# X['Bohemian Index'] = X['Bohemian Index']**(1/2)
# X['Talent Index'] = X['Talent Index']**(1/2)
# X['Proportion of Young People'] = X['Proportion of Young People']**(1/2)
# X['Unemployment Ratio'] = X['Unemployment Ratio']**(1/2)
# X['Poverty by Income Percentage'] = X['Poverty by Income Percentage']**(1/2)
# X['Median Household Income'] = X['Median Household Income']**(1/2)
# X['Median Household Value'] = X['Median Household Value']**(1/2)
# X['Proportion of Owner Occupied Residences'] = X['Proportion of Owner Occupied Residences']**(1/2)
X['Number of Hotels'] = X['Number of Hotels']**(1/2)
# X['Bus Stops'] = X['Bus Stops']**(1/2)
X['Point of Interests'] = X['Point of Interests']**(1/2)
# X['Distance to Center'] = X['Distance to Center']**(1/2)
X['Airbnb penetration'] = X['Airbnb penetration']**(1/2)

print("Skewness after")
print(X.skew(axis=0))

#Normalise data
X = ((X-X.mean())/X.std())

#VIF statistics
Y = X['Airbnb penetration']
X = X[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']]

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("VIF")
print(vif_data)

# Regression plot
x = sm.add_constant(X)
model = sm.OLS(Y,x).fit()
predictions = model.predict(x)
print("Regression", model.summary())
print("coefficients")
for coef in list(model.params):
    print(":", coef)
print("p-values")
for i, coef in enumerate(list(model.params)):
    print(":", model.pvalues[i])

# Calculate Moran's I
with open('coordinates_center_per_tract.txt', 'r') as f:
    coordinates_center = json.loads(f.read())
tracts_coordinates = coordinates_center[city]

# Remove tracts which are not in csv city file
tracts_dictionary = list(tracts_coordinates.keys())
for tract in tracts_dictionary:
    if tract not in tracts:
        tracts_coordinates.pop(tract, None)

connectivity_matrix = np.empty((len(tracts_coordinates)+1, len(tracts_coordinates)+1), dtype=object)
connectivity_matrix[1:,0] = list(tracts_coordinates.keys())
connectivity_matrix[0,1:] = list(tracts_coordinates.keys())
for i in tqdm(range(1, len(tracts_coordinates)+1)):
    for j in range(1, len(tracts_coordinates)+1):
        if i == j:
            connectivity_matrix[i,j] = 0
        elif connectivity_matrix[i,j] is None:
            # Calculate inverse Euclidean distance
            euclidean_distance = 1 / math.dist(tracts_coordinates[connectivity_matrix[i,0]], tracts_coordinates[connectivity_matrix[0,j]])
            connectivity_matrix[i,j] = euclidean_distance
            connectivity_matrix[j,i] = euclidean_distance

geographical = ['Distance to Center','Point of Interests', 'Number of Hotels', 'Bus Stops', 'Population Density']

y_features = X[geographical]
for feature in geographical:
    mean_feature = X[feature].mean()
    y_features[feature] -= mean_feature
cross_product = np.prod(y_features, axis=1)
moran = Moran(cross_product, full2W(connectivity_matrix[1:,1:]))
print("Moran's I")
print(":", moran.I)
