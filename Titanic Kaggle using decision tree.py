# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:54:56 2019

@author: mynam
"""

import pandas as pd
test = pd.read_csv("test.csv")
test_shape = test.shape
print(test_shape)

train = pd.read_csv("train.csv")
train_shape = train.shape
print(train_shape)

import matplotlib.pyplot as plt


#Preview the first few rows of data
train.head()
test.head()

#Draft up a pivot table
sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot

#look into class
pclass_pivot = train.pivot_table(index="Pclass",values="Survived")
# pclass_pivot, Draw a bar chart to visualise effect of class on survival
pclass_pivot.plot.bar()
plt.show()

#details of the ages of the passengers. Gives averages and quartiles. 
train['Age'].describe()

#table of people who survived 
train[train["Survived"] == 1]

#comparison of distribution of ages of survivors vs non-survivors
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

#Cut the Ages into different sections. 
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

age_cat_pivot = train.pivot_table(index="Age_categories",values="Survived")
age_cat_pivot.plot.bar()
plt.show()

#Count no of passengers in each class
train['Pclass'].value_counts()

#Apply dummy variables to the age class. 
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")
train.head()

#Create dummies for Sex and Age categories
train = create_dummies(train,"Sex")
test = create_dummies(test,"Sex")
train = create_dummies(train,"Age_categories")
test = create_dummies(test,"Age_categories")

train.head()



holdout = test # from now on we will refer to this
               # dataframe as the holdout data to avoid any confusion


#Split the dataset into a test set and train set. 80% training, 20% testing


from sklearn.model_selection import train_test_split

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.2,random_state=0)

#Check shape of data
train_X.shape




"""
#Creat Logistic Regression object
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

#Create K-nearest neighbours object
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()


#Create the SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
"""
"""
#Fitting Naive Bayes to the model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
"""


#Fitting decision tree to the Training Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)


#Fit classifier
classifier.fit(train_X, train_y)
predictions = classifier.predict(test_X)



#Check the accuracy of the fitted model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions)
print(accuracy)

#Create a confusion matrix to check the accuracy of the model
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(test_y, predictions)
pd.DataFrame(conf_matrix, columns=['Survived', 'Died'], index=[['Survived', 'Died']])

#Use K-fold cross validation to mix up the training and test data and repeat the tests
from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(lr, all_X, all_y, cv=10)
print(scores)
np.mean(scores)
#This is over 80% Not too bad. Let's fit!


#Fit the logistic regression to the 'holdout' dataset. 
holdout.head()
classifier.fit(all_X, all_y)
holdout_predictions = classifier.predict(holdout[columns])
holdout_predictions


#Create and export the submission dataframe.
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv('titanic_submission_Decision_tree.csv', index=False)




