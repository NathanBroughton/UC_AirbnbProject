# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:10:10 2022

@author: NatBr
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os
import time
from collections import Counter
import json
import math
from tqdm import tqdm
from esda.moran import Moran
from libpysal.weights import full2W


pd.options.mode.chained_assignment = None

ROS = RandomOverSampler()
RUS = RandomUnderSampler()
SM = SMOTE()
n = 1


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
        # If no space in city name
        city = str(file[:-4])
        # If space in city name
        # city = "New York"
        
df_train = pd.concat(train_set,axis=0,ignore_index=True)
df_test = pd.concat(test_set,axis=0,ignore_index=True)

df_all = pd.concat([df_train,df_test],axis=0,ignore_index=True)
train_shape = df_train.shape

tracts = list(df_test["NAME"])

#Data Description
df_all = df_all.drop(['Unnamed: 0','state','county','tract','Tracts'],axis=1)
print(df_all.info())

#Plot penetration count histogram
Y = df_all['Airbnb penetration']
w = Counter(Y)

#Convert penetration number to class
df_all.to_csv('df_all.csv')
smallest = df_all.nsmallest(int(0.7*len(df_all)),['Airbnb penetration']).index
largest = df_all.nlargest(int(0.05*len(df_all)),['Airbnb penetration']).index
#largest = df_all.nlargest(int(0.1*len(df_all)),['Airbnb penetration']).index

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

#Adjust skewness print("Skewness before")
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

#Classification (Comment to adjust for skewness)
    
df_all['Number of Hotels'] = df_all['Number of Hotels']**(1/2)
#df_all['Population Density'] = df_all['Population Density']**(1/2)
#df_all['Proportion of Owner Occupied Residences'] = df_all['Proportion of Owner Occupied Residences']**(1/2)
#df_all['Bus Stops'] = df_all['Bus Stops']**(1/2)
df_all['Point of Interests'] = df_all['Point of Interests']**(1/2)
df_all['Airbnb penetration'] = df_all['Airbnb penetration']**(1/2)

#Paper sets
set0 = ['Population Density','Bohemian Index', \
        'Talent Index','Proportion of Young People' \
        ,'Median Household Income','Median Household Value' \
        ,'Point of Interests','Distance to Center']
    
set1 = ['Population Density','Bohemian Index','Number of Hotels','Bus Stops','Distance to Center']

X_train = df_all[set0][:train_shape[0]]
    
y_train = df_all['Airbnb penetration_class'][:train_shape[0]]

X_test = df_all[set0][train_shape[0]:]
    
y_test = df_all['Airbnb penetration_class'][train_shape[0]:]

X_train = ((X_train-X_train.mean())/X_train.std())
X_test = ((X_test-X_test.mean())/X_test.std())

Train = []
for i in range(n): #(Comment for US, SM or nothing)
    #X_train_app, y_train_app = X_train,y_train
    X_train_app, y_train_app = RUS.fit_resample(X_train,y_train)
    #X_train_app, y_train_app = SM.fit_resample(X_train,y_train)
    Train.append(X_train_app)
    Train.append(y_train_app)
#X_train, y_train = RUS.fit_resample(X_train,y_train)
#X_train, y_train = SM.fit_resample(X_train,y_train)

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

def run_model(model):
    #for i in range(n):
    predictions = []
    i = 0
    while i < n*2:
        clf = model.fit(Train[i],Train[i+1])
        y_pred = clf.predict(X_test)
        predictions.append(y_pred)
        i += 2
    y_pred = []
    for i in range(len(predictions[0])):
        y = []
        for j in range(n):
            y.append(predictions[j][i])
        y_pred.append(max(set(y),key = y.count))
    cm = metrics.confusion_matrix(y_test, y_pred)
    l,m,h = evaluate(cm)   
    
    return l,m,h

time1 = time.time()

l,m,h = run_model(svm.SVC())

print("SVM Accuracy:", (l+m+h)/3)
print("SVM took:", time.time() - time1)

time2 = time.time()
l,m,h = run_model(LogisticRegression(solver='liblinear',random_state=0))

print("Logistic Accuracy:", (l+m+h)/3)
print("Logistic took:", time.time() - time2)

time3 = time.time()
l,m,h = run_model(RandomForestClassifier(random_state=42))

print("Random Forest Accuracy",(l+m+h)/3)
print("Random Forest took:", time.time() - time3)

time4 = time.time()
l,m,h = run_model(GaussianNB())

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