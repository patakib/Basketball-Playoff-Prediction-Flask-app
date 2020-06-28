# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 07:53:50 2020

@author: patak
"""

#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle

#import dataset
df = pd.read_csv('nbaallelo.csv')

#feature engineering
df['TeamYear'] = df['year_id'].astype(str) + '-' + df['team_id']
df_playoff = df[df['is_playoffs']==1]
        
playoffteams = df_playoff['TeamYear'].unique().tolist()
allteams = df['TeamYear'].unique().tolist()
labels = list()
        
for team in allteams:
    if team in playoffteams:
        labels.append(1)
    else:
        labels.append(0)
        
teamlabels = pd.DataFrame(
    {'TeamYear': allteams,
     'Playoff': labels
         })
        
datacolumns = ["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10"]
        # ,"M11","M12","M13","M14","M15","M16","M17","M18","M19","M20"
for column in datacolumns:
    teamlabels[column] = np.nan
        
        
teamlabels = teamlabels.set_index('TeamYear')
        
for team in allteams:
    teamrecord = df[(df['TeamYear']==team) & (df['is_playoffs']==0)]
    teamrecord = teamrecord[['game_result']]
    teamrecord = teamrecord.head(10)
    teamrecord = teamrecord.rename(columns={'game_result': team})
    teamrecord = teamrecord.T
    teamrecord.columns = datacolumns
    teamlabels.update(teamrecord)
    lst = [teamrecord]
    del lst
          
y = teamlabels[['Playoff']]
teamlabels = teamlabels.drop(['Playoff'], axis=1)
X = teamlabels.copy()
X = pd.get_dummies(X)
X['Win']=X['M1_W']+X['M2_W']+X['M3_W']+X['M4_W']+X['M5_W']+X['M6_W']+X['M7_W']+X['M8_W']+X['M9_W']+X['M10_W']
X['Lost']=X['M1_L']+X['M2_L']+X['M3_L']+X['M4_L']+X['M5_L']+X['M6_L']+X['M7_L']+X['M8_L']+X['M9_L']+X['M10_L']
X=X[['Win','Lost']]

#split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# log_clf.fit(X_train, y_train.values.ravel())
# y_pred = log_clf.predict(X_test)
# print(confusion_matrix(y_test.values.ravel(),y_pred))
# print("Precision: {:.2f}%".format(100 * precision_score(y_test.values.ravel(),y_pred)))
# print("Recall: {:.2f}%".format(100 * recall_score(y_test.values.ravel(),y_pred)))
# print("F1: {:.2f}%".format(100 * f1_score(y_test.values.ravel(),y_pred)))

# svc_clf = SVC(gamma='auto')
# svc_clf.fit(X_train, y_train.values.ravel())
# y_pred = svc_clf.predict(X_test)
# print(confusion_matrix(y_test.values.ravel(),y_pred))
# print("Precision: {:.2f}%".format(100 * precision_score(y_test.values.ravel(),y_pred)))
# print("Recall: {:.2f}%".format(100 * recall_score(y_test.values.ravel(),y_pred)))
# print("F1: {:.2f}%".format(100 * f1_score(y_test.values.ravel(),y_pred)))

# tree_clf = DecisionTreeClassifier(max_depth=5)
# tree_clf.fit(X_train, y_train.values.ravel())
# y_pred = tree_clf.predict(X_test)
# print(confusion_matrix(y_test.values.ravel(),y_pred))
# print("Precision: {:.2f}%".format(100 * precision_score(y_test.values.ravel(),y_pred)))
# print("Recall: {:.2f}%".format(100 * recall_score(y_test.values.ravel(),y_pred)))
# print("F1: {:.2f}%".format(100 * f1_score(y_test.values.ravel(),y_pred)))

#fit the dataset into Random Forest Classifier algorithm
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train.values.ravel())
# y_pred = rnd_clf.predict(X_test)
# print(confusion_matrix(y_test.values.ravel(),y_pred))
# print("Precision: {:.2f}%".format(100 * precision_score(y_test.values.ravel(),y_pred)))
# print("Recall: {:.2f}%".format(100 * recall_score(y_test.values.ravel(),y_pred)))
# print("F1: {:.2f}%".format(100 * f1_score(y_test.values.ravel(),y_pred)))

#checkpoint for the predictor
# print(rnd_clf.predict([[4,6]]))

# Saving model to disk
pickle.dump(rnd_clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[8, 2]]))


