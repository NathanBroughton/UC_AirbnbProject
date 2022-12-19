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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.ticker import ScalarFormatter
import os
import time
import matplotlib.pyplot as plt
from collections import Counter

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
for file in os.listdir('./Data3/test'):
        path = os.path.join('./Data3/test',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        test_set.append(df)
        
df_train = pd.concat(train_set,axis=0,ignore_index=True)
df_test = pd.concat(test_set,axis=0,ignore_index=True)

print(df_train.shape)
print(df_test.shape)
df_all = pd.concat([df_train,df_test],axis=0)
print(df_all.shape)
train_shape = df_train.shape
print(train_shape[0])

#Plot penetration count histogram
Y = df_all['Airbnb penetration']
w = Counter(Y)
print(len(w))
fig, ax = plt.subplots()
plt.hist(Y**(1/2), bins=40, edgecolor='black', linewidth=0.5)
# plt.bar(w.keys(),w.values())
plt.xlabel(r"$\sqrt{bnb\_penetration}$")
plt.ylabel("count")
plt.title("Frequency distribution of Airbnb penetration")
# plt.xscale('log')
plt.yscale('log')
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
plt.show()

#Convert penetration number to class
smallest = df_all.nsmallest(int(0.7*len(df_all)),['Airbnb penetration']).index
largest = df_all.nlargest(int(0.1*len(df_all)),['Airbnb penetration']).index

df_all['Airbnb penetration_class'] = [1]*len(df_all)
for i in smallest:
    #print(df_all['Airbnb penetration_class'][i])
    df_all['Airbnb penetration_class'][i] = 0

for i in largest:
    #print(df_all['Airbnb penetration_class'][i])
    df_all['Airbnb penetration_class'][i] = 2

#Start explanotary analysis
X = df_test[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center','Airbnb penetration']]

y = df_all['Airbnb penetration_class']

#Adjust skewness
X['Number of Hotels'] = X['Number of Hotels']**(1/2)
X['Population Density'] = X['Population Density']**(1/2)
X['Point of Interests'] = X['Point of Interests']**(1/2)
#X['Distance to Center'] = X['Distance to Center']**(1/2)
X['Airbnb penetration'] = X['Airbnb penetration']**(1/2)

#X['Number of Hotels'] = np.log(X['Number of Hotels'].replace(0, np.nan))
#X['Distance to Center'] = np.log(X['Distance to Center'])
#X['Point of Interests'] = np.log(X['Point of Interests'].replace(0, np.nan))
#X['Number of Hotels'] = X['Number of Hotels'].replace(np.nan,0)
#X['Point of Interests'] = X['Point of Interests'].replace(np.nan,0)


print(X.skew(axis=0))
#Normalise data
X = ((X-X.mean())/X.std())

#VIF statistics
Y = X['Airbnb penetration']
X = X[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']]
#print(df_all['Airbnb penetration'].skew(axis=0))
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data) 

#Regression plot
x = sm.add_constant(X)
model = sm.OLS(Y,x).fit()
predictions = model.predict(x)
print(model.summary())

#Classification
    
df_all['Number of Hotels'] = df_all['Number of Hotels']**(1/2)
df_all['Population Density'] = df_all['Population Density']**(1/2)
df_all['Point of Interests'] = df_all['Point of Interests']**(1/2)
#X['Distance to Center'] = X['Distance to Center']**(1/2)
df_all['Airbnb penetration'] = df_all['Airbnb penetration']**(1/2)

#Paper sets
set0 = ['Race Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Poverty by Income Percentage' \
        ,'Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']

set1 = ['Population Density','Race Diversity Index','Income Diversity Index', \
        'Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']
    
set2 = ['Population Density','Race Diversity Index','Income Diversity Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio' \
        ,'Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']
    
set3 = ['Population Density','Race Diversity Index','Income Diversity Index', \
        'Proportion of Young People','Unemployment Ratio' \
        ,'Median Household Income','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']
    
set4 = ['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Proportion of Young People','Unemployment Ratio' \
        ,'Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']
    
set5 = ['Population Density','Race Diversity Index','Income Diversity Index', \
        'Proportion of Young People','Unemployment Ratio' \
        ,'Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']

X_train = df_all[set0][:train_shape[0]]
    
y_train = df_all['Airbnb penetration_class'][:train_shape[0]]

X_test = df_all[set0][train_shape[0]:]
    
y_test = df_all['Airbnb penetration_class'][train_shape[0]:]

X_train = ((X_train-X_train.mean())/X_train.std())
X_test = ((X_test-X_test.mean())/X_test.std())

print(X_train.head())

print(X_train.shape)
print(y_train.shape)

X_train, y_train = SM.fit_resample(X_train,y_train)

time1 = time.time()

clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("SVM Accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print("SVM took:", time.time() - time1)

time2 = time.time()

clf = LogisticRegression(solver='liblinear',random_state=0, class_weight={0:1,1:10,2:50}).fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Logistic Accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))
print("Logistic took:", time.time() - time2)

time3 = time.time()

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Random Forest Accuracy", metrics.balanced_accuracy_score(y_test,y_pred))
print("Random Forest took:", time.time() - time3)

time4 = time.time()

clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Naive Bayes Accuracy", metrics.balanced_accuracy_score(y_test,y_pred))
print("Naive Bayes took:", time.time() - time4)