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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import time
import matplotlib.pyplot as plt
from collections import Counter

pd.options.mode.chained_assignment = None

ROS = RandomOverSampler()
RUS = RandomUnderSampler()

train = True
i = 1
train_set = []
test_set = []
for file in os.listdir('./Data2'):
    if(i == 21):
        train = False
    if(train):
        path = os.path.join('./Data2',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        train_set.append(df)
        i += 1
    else:
        print(file)
        path = os.path.join('./Data2',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        test_set.append(df)
        
df_train = pd.concat(train_set,axis=0,ignore_index=True)
df_test = pd.concat(test_set,axis=0,ignore_index=True)

print(df_train.shape)
print(df_test.shape)
df_all = pd.concat([df_train,df_test],axis=0)
print(df_all.shape)

Y = df_all['Airbnb penetration']
w = Counter(Y)
plt.bar(w.keys(),w.values())
plt.show()

print(Counter(Y))
#print((Y<20).sum())
#plt.hist(Y,bins=np.logspace(np.log10(1),np.log10(3),200))
#plt.gca().set_xscale("log")
#plt.show()

smallest = df_all.nsmallest(int(0.7*len(df_all)),['Airbnb penetration']).index
largest = df_all.nlargest(int(0.1*len(df_all)),['Airbnb penetration']).index

df_all['Airbnb penetration_class'] = [1]*len(df_all)
for i in smallest:
    #print(df_all['Airbnb penetration_class'][i])
    df_all['Airbnb penetration_class'][i] = 0

for i in largest:
    #print(df_all['Airbnb penetration_class'][i])
    df_all['Airbnb penetration_class'][i] = 2

#print(df_train.columns)
#print(df_train.head())


X = df_all[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests','Distance to Center']]

y = df_all['Airbnb penetration_class']

X['Number of Hotels'] = np.log(X['Number of Hotels'].replace(0, np.nan))
#X['Distance to Center'] = np.log(X['Distance to Center'])
X['Point of Interests'] = np.log(X['Point of Interests'].replace(0, np.nan))
X['Number of Hotels'] = X['Number of Hotels'].replace(np.nan,0)
X['Point of Interests'] = X['Point of Interests'].replace(np.nan,0)

print(X.skew(axis=0))
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data) 

#x = df[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', 'Talent Index','Proportion of Young People','Unemployment Ratio', \'Proportion of Owner Occupied Residences','Number of Hotels','Bus Stops']]
#print(x)
x = X[['Population Density','Bohemian Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage','Median Household Value','Number of Hotels','Bus Stops','Point of Interests','Distance to Center']]
x = sm.add_constant(x)
model = sm.OLS(Y,x).fit()
predictions = model.predict(x)
print(model.summary())

        
X_train = X[:5025]
    

    
y_train = y[:5025]

X_test = X[5025:]
    
y_test = y[5025:]


X_train = ((X_train-X_train.mean())/X_train.std())
X_test = ((X_test-X_test.mean())/X_test.std())

print(X_train.head())

print(X_train.shape)
print(y_train.shape)

#X_train, y_train = RUS.fit_resample(X_train,y_train)

#print(Counter(y_train))


"""#print(asfgds)

#df = pd.read_csv("Data.csv")

x = df[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests']]
y = df['Airbnb penetration']

print(df.skew(axis=0)) #Return acceptable skewness ranges -3 to 3, so now log needed (DOUBLE CHECK)

#print(x)
#x = np.log(x)
#print(x)
x = ((x-x.mean())/x.std())
print(x.columns)
print(x.values)
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data) 
#x = df[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', 'Talent Index','Proportion of Young People','Unemployment Ratio', \'Proportion of Owner Occupied Residences','Number of Hotels','Bus Stops']]
#print(x)
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
predictions = model.predict(x)

print(model.summary())



    
print(df['Airbnb penetration_class'])
print(df.dtypes)

train = df[:20]
test = df[20:]

X_train = train[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests']]
    
y_train = train['Airbnb penetration_class']

X_test = test[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests']]
    
y_test = test['Airbnb penetration_class']

"""

#print(int(0.*len(df)))
#print(int(0.1*len(df)))

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











