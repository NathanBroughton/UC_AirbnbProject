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
import os

train = True
i = 1
train_set = []
test_set = []
for file in os.listdir('./Data'):
    if(i == 21):
        train = False
    if(train):
        path = os.path.join('./Data',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        train_set.append(df)
        i += 1
    else:
        print(file)
        path = os.path.join('./Data',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        test_set.append(df)
        
df_train = pd.concat(train_set,axis=0,ignore_index=True)
df_test = pd.concat(test_set,axis=0,ignore_index=True)

print(df_train.shape)
print(df_test.shape)
df_all = pd.concat([df_train,df_test],axis=0)
print(df_all.shape)

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
        
X_train = df_all[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests']][:5025]
    

    
y_train = df_all['Airbnb penetration_class'][:5025]

X_test = df_all[['Population Density','Race Diversity Index','Income Diversity Index','Bohemian Index', \
        'Talent Index','Proportion of Young People','Unemployment Ratio','Poverty by Income Percentage' \
        ,'Median Household Income','Median Household Value','Proportion of Owner Occupied Residences' \
        ,'Number of Hotels','Bus Stops','Point of Interests']][5025:]
    
y_test = df_all['Airbnb penetration_class'][5025:]

print(X_train.shape)
print(y_train.shape)


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

clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("SVM Accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))

clf = LogisticRegression(solver='liblinear',random_state=0).fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Logistic Accuracy:", metrics.balanced_accuracy_score(y_test,y_pred))

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Random Forest Accuracy", metrics.balanced_accuracy_score(y_test,y_pred))

clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Naive Bayes Accuracy", metrics.balanced_accuracy_score(y_test,y_pred))









