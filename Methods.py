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


pd.options.mode.chained_assignment = None

ROS = RandomOverSampler()
RUS = RandomUnderSampler()
SM = SMOTE()


#Initiate training and test set
train_set = []
test_set = []
for file in os.listdir('./Data3/Train'):
        path = os.path.join('./Data3/Train',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        train_set.append(df)
for file in os.listdir('./Data3/Test'):
        path = os.path.join('./Data3/Test',file)
        df = pd.read_csv(path,index_col=None,header = 0)
        test_set.append(df)
        
df_train = pd.concat(train_set,axis=0,ignore_index=True)
df_test = pd.concat(test_set,axis=0,ignore_index=True)

print(df_train.shape)
print(df_test.shape)
df_all = pd.concat([df_train,df_test],axis=0,ignore_index=True)
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
print(df_all.nlargest(int(0.1*len(df_all)),['Airbnb penetration']).index)
df_all.to_csv('df_all.csv')
smallest = df_all.nsmallest(int(0.7*len(df_all)),['Airbnb penetration']).index
largest = df_all.nlargest(int(0.05*len(df_all)),['Airbnb penetration']).index
#largest = df_all.nlargest(int(0.1*len(df_all)),['Airbnb penetration']).index

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
set0 = ['Population Density','Bohemian Index', \
        'Talent Index','Proportion of Young People' \
        ,'Median Household Income','Median Household Value' \
        ,'Point of Interests','Distance to Center']
    
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

X_train1, y_train1 = RUS.fit_resample(X_train,y_train)
X_train2, y_train2 = RUS.fit_resample(X_train,y_train)
X_train3, y_train3 = RUS.fit_resample(X_train,y_train)
X_train4, y_train4 = RUS.fit_resample(X_train,y_train)
X_train5, y_train5 = RUS.fit_resample(X_train,y_train)

def evaluate(cm):
    #Low
    c_low = sum(cm[1]+cm[2])/sum(cm[0])
    print(c_low)
    TP_low = cm[0][0]
    FP_low = cm[1][0] + cm[2][0]
    FN_low = cm[0][1] + cm[0][2] 
    TN_low = cm[1][1] + cm[2][2] + cm[1][2] + cm[2][1]

    WA_low = ((c_low*TP_low)+TN_low)/((c_low*(TP_low + FN_low))+(TN_low + FP_low))
    print(WA_low)
    
    #Medium
    c_med = sum(cm[0]+cm[2])/sum(cm[1])
    TP_med = cm[1][1]
    FP_med = cm[0][1] + cm[2][1]
    FN_med = cm[1][0] + cm[1][2] 
    TN_med = cm[0][0] + cm[2][2] + cm[0][2] + cm[2][0]
    
    WA_med = ((c_med*TP_med)+TN_med)/((c_med*(TP_med + FN_med))+(TN_med + FP_med))
    print(WA_med)
    #High
    
    c_high = sum(cm[0]+cm[1])/sum(cm[2])
    TP_high = cm[2][2]
    FP_high = cm[0][2] + cm[1][2]
    FN_high = cm[2][0] + cm[2][1] 
    TN_high = cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]

    WA_high = ((c_high*TP_high)+TN_high)/((c_high*(TP_high + FN_high))+(TN_high + FP_high))
    print(WA_high)
    
    return WA_low, WA_med, WA_high

time1 = time.time()

clf1 = svm.SVC().fit(X_train1,y_train1)
clf2 = svm.SVC().fit(X_train2,y_train2)
clf3 = svm.SVC().fit(X_train3,y_train3)
clf4 = svm.SVC().fit(X_train4,y_train4)
clf5 = svm.SVC().fit(X_train5,y_train5)
#clf = OneVsRestClassifier(svm.SVC()).fit(X_train,y_train)
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred4 = clf3.predict(X_test)
y_pred5 = clf3.predict(X_test)
y_pred = []
#class_0, class_1, class_2 = 0
for i in range(len(y_pred1)):
    y = [y_pred1[i],y_pred2[i],y_pred3[i],y_pred4[i],y_pred5[i]]
    #print(y_pred1[i],y_pred2[i],y_pred3[i])
    #print(max(set(y),key = y.count))
    y_pred.append(max(set(y),key = y.count))
    
#print(metrics.accuracy_score(y_test,y_pred))

cm = metrics.confusion_matrix(y_test, y_pred)
l,m,h = evaluate(cm)


print(metrics.confusion_matrix(y_test, y_pred))
print("SVM Accuracy:", (l+m+h)/3)
print("SVM took:", time.time() - time1)

time2 = time.time()

clf1 = LogisticRegression(solver='liblinear',random_state=0).fit(X_train1,y_train1)
clf2 = LogisticRegression(solver='liblinear',random_state=0).fit(X_train2,y_train2)
clf3 = LogisticRegression(solver='liblinear',random_state=0).fit(X_train3,y_train3)
clf4 = LogisticRegression(solver='liblinear',random_state=0).fit(X_train4,y_train4)
clf5 = LogisticRegression(solver='liblinear',random_state=0).fit(X_train5,y_train5)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred4 = clf4.predict(X_test)
y_pred5 = clf5.predict(X_test)

y_pred = []
#class_0, class_1, class_2 = 0
for i in range(len(y_pred1)):
    y = [y_pred1[i],y_pred2[i],y_pred3[i],y_pred4[i],y_pred5[i]]
    #print(y_pred1[i],y_pred2[i],y_pred3[i])
    #print(max(set(y),key = y.count))
    y_pred.append(max(set(y),key = y.count))

cm = metrics.confusion_matrix(y_test, y_pred)
l,m,h = evaluate(cm)

print("Logistic Accuracy:", (l+m+h)/3)
print("Logistic took:", time.time() - time2)

time3 = time.time()

clf1 = RandomForestClassifier(random_state=42).fit(X_train1,y_train1)
clf2 = RandomForestClassifier(random_state=42).fit(X_train2,y_train2)
clf3 = RandomForestClassifier(random_state=42).fit(X_train3,y_train3)
clf4 = RandomForestClassifier(random_state=42).fit(X_train4,y_train4)
clf5 = RandomForestClassifier(random_state=42).fit(X_train5,y_train5)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred4 = clf4.predict(X_test)
y_pred5 = clf5.predict(X_test)

y_pred = []
#class_0, class_1, class_2 = 0
for i in range(len(y_pred1)):
    y = [y_pred1[i],y_pred2[i],y_pred3[i],y_pred4[i],y_pred5[i]]
    #print(y_pred1[i],y_pred2[i],y_pred3[i])
    #print(max(set(y),key = y.count))
    y_pred.append(max(set(y),key = y.count))

cm = metrics.confusion_matrix(y_test, y_pred)
l,m,h = evaluate(cm)

print("Random Forest Accuracy",(l+m+h)/3)
print("Random Forest took:", time.time() - time3)

time4 = time.time()

clf1 = GaussianNB().fit(X_train1,y_train1)
clf2 = GaussianNB().fit(X_train2,y_train2)
clf3 = GaussianNB().fit(X_train3,y_train3)
clf4 = GaussianNB().fit(X_train4,y_train4)
clf5 = GaussianNB().fit(X_train5,y_train5)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)
y_pred4 = clf4.predict(X_test)
y_pred5 = clf5.predict(X_test)

y_pred = []
#class_0, class_1, class_2 = 0
for i in range(len(y_pred1)):
    y = [y_pred1[i],y_pred2[i],y_pred3[i],y_pred4[i],y_pred5[i]]
    #print(y_pred1[i],y_pred2[i],y_pred3[i])
    #print(max(set(y),key = y.count))
    y_pred.append(max(set(y),key = y.count))

cm = metrics.confusion_matrix(y_test, y_pred)
l,m,h = evaluate(cm)

print("Naive Bayes Accuracy", (l+m+h)/3)
print("Naive Bayes took:", time.time() - time4)

#Ensemble
time5 = time.time()

clf1 = svm.SVC()
clf2 = LogisticRegression(solver='liblinear',random_state=0)
clf3 = RandomForestClassifier(random_state=42)
clf4 =  GaussianNB()

final_model = VotingClassifier(estimators=[('svm',clf1),('lr',clf2),('rf',clf3),('g',clf4)],voting='hard')
final_model.fit(X_train,y_train)
y_pred = final_model.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
l,m,h = evaluate(cm)

print("Ensemble Accuracy", (l+m+h)/3)
print("Ensemble took:", time.time() - time5)