#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', '')
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import *
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable
from astropy.table import Table, Column
import pickle
from sklearn.externals import joblib 


# In[2]:


gender_train = pd.read_csv("train.csv")
#gender_train
gender_train.groupby("gender").gender.count().plot(kind = 'bar')


# In[3]:


gender_test = pd.read_csv("test.csv")
gender_test.groupby("gender").gender.count().plot(kind = 'bar')


# In[4]:


hair_train = gender_train.groupby("hair").hair.count().plot(kind = "bar")


# In[5]:


hair_test = gender_test.groupby("hair").hair.count().plot(kind = "bar")


# In[6]:


beard_train = gender_train.groupby("beard").beard.count().plot(kind = "bar")


# In[7]:


beard_test = gender_test.groupby("beard").beard.count().plot(kind = "bar")


# In[8]:


encoded_train = gender_train.iloc[:,2:6].apply(LabelEncoder().fit_transform)
gender_train_reduced = gender_train.drop(columns = ["hair","beard","scarf","gender"])
gender_train_reduced
encoded_train


# In[9]:


X1 = pd.concat([gender_train_reduced,encoded_train],axis = 1).round(2)
X1


# In[10]:


encoded_test = gender_test.iloc[:,2:6].apply(LabelEncoder().fit_transform)
gender_test_reduced = gender_test.drop(columns = ["hair", "beard", "scarf", "gender"])
gender_test_reduced


# In[11]:


X2 = pd.concat([gender_test_reduced, encoded_test], axis = 1).round(2)
X2


# In[12]:


#X = gender_train.drop(columns = ["gender"])
#y = encoded_train["gender"]
#print(y_tr
#LogReg = LogisticRegression()
#LogReg.fit(encoded_train, y_train)
#prediction = LogReg.predict(encoded_test)
X_train = X1.drop(columns = ["gender"])
X_train
y_train = X1["gender"]
y_train
X_test = X2.drop(columns = ["gender"])
y_test = X2["gender"]
y_test

LogReg = LogisticRegression(solver = "lbfgs")
LogReg.fit(X_train, y_train)
predLog = LogReg.predict(X_test)

logScore = accuracy_score(y_test,predLog)
print(f"The score of accuracy for Logistic Regression is {100*logScore} Percent")


# In[13]:


randomForest = RandomForestClassifier(n_estimators = 20)
randomForest.fit(X_train, y_train)
predForest = randomForest.predict(X_test)

forestScore = accuracy_score(y_test, predForest)
print(f"The score of accuracy for RandomForest is {forestScore*100} Percent")


# In[14]:


linear = LinearSVC(max_iter = 2000)
linear.fit(X_train, y_train)
predLinear = linear.predict(X_test)

linearScore = accuracy_score(y_test, predLinear)
print(f"The score of linearSvc is {100*linearScore} Percent")


# In[15]:


bernouli = BernoulliNB()
bernouli.fit(X_train, y_train)
predBer = bernouli.predict(X_test)
bernouliScore = accuracy_score(y_test, predBer)
print(f"The score of BernoulliNB is {100*bernouliScore} Percent")


# In[16]:


print("Detailed Performance of all Models: ")
print("=====================================")
print("+-------------------------+---------+")
print("|          Model        |  Accuracy |")
print(f"|  LogisticRegression   |    {logScore}   |")
print(f"|RandomForestClassifier |    {forestScore}   | ")
print(f"|      LinearSvc        |    {linearScore}   | ")
print(f"|      BernoulliNB      |    {bernouliScore}    |")
print("+-------------------------+---------+")


# In[17]:


print()
print("Best Model.")
print("=====================================")
print("+-------------------------+---------+")
print("|          Model        |  Accuracy |")
print("+-------------------------+---------+")
print(f"|      BernoulliNB      |    {bernouliScore}    |")
print("+-------------------------+---------+")


# In[18]:


combine = pd.concat([X1,X2])
combine
#gender_test


# In[19]:


X = combine.drop(columns = ["gender"])
X
y = combine["gender"]
y
bernouli.fit(X, y)

filename = 'finalized_model.sav'
joblib.dump(bernouli, filename)

prediction = bernouli.predict(X)
score = accuracy_score(y, prediction)
print(f"The score of best model for accuracy is {score*100} Percent")


# In[68]:


height = int(input("Enter your Height(centimeter): "))
weight = int(input("Enter your weight(kg): "))
hair = input("Enter lenght of your hair (Long, Meduium, Short, Bald)")
beard = input("Do you have beard? (Yes/No): ")
scarf = input("Do you wear Scarf? (Yes/NO): ")

#userInput = pd.DataFrame({"Height": height, "Weight": weight, "HairSHo hair, "Beard": beard, "Scarf": scarf}, index = [0])
#userInput
data = [[height,weight,hair,beard,scarf]]
data = pd.DataFrame(data, columns = ["Height","Weight","Hair","Beard","Scarf"])


# In[69]:


hairMapper = {'Bald': 0,
              'Long':1,
              'Medium':2,
              'Short':3}
beardMapper = {
                'Yes': 1,
                'No': 0}

scarfMapper = {
                'No': 0,
                'Yes': 1}


data["Scarf"] = data["Scarf"].replace(scarfMapper)

data["Beard"] = data["Beard"].replace(beardMapper)

data["Hair"] = data["Hair"].replace(hairMapper)
#userInput= userInput.drop(columns = ["HairValuse"])# Ignore this line 


# In[70]:


data


# In[71]:


loaded_model = joblib.load(filename)


# In[72]:


prediction = loaded_model.predict(data)
prediction = int(prediction[0])
print(prediction)
genderMapper = {
                1: 'Male',
                0: 'Female'}
prediction = [[score]]
prediction = pd.DataFrame(output, columns = ["Gender"])
prediction.replace(genderMapper)


# In[66]:




