# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:57:59 2019

@author: mynam
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

train_dataset = pd.read_csv('train.csv')
X_train = train_dataset.iloc[:, [2,4,5,6,7,9,11]].values
y = train_dataset.iloc[:, 1].values

test_dataset = pd.read_csv('test.csv')
X_test = test_dataset.iloc[:, [1,3,4,5,6,8,10]].values




#Taking care of missing values
from sklearn.impute import SimpleImputer
imputer1 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer1.fit(X_train[:, [0,1,6]])
X_train[:, [0,1,6]] = imputer.transform(X_train[:, [0,1,6]])

imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer2.fit(X_train[:, [2,3,4,5]])
X_train[:, [2,3,4,5]] = imputer.transform(X_train[:, [2,3,4,5]])



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
imputer.fit(X_train[:, [0,6]])
X_train[:, [0,2,3,4,5]] = imputer.transform(X_train[:, [0,2,3,4,5]])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X_test[:, [0,2,3,4,5]])
X_test[:, [0,2,3,4,5]] = imputer.transform(X_test[:, [0,2,3,4,5]])



#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_train = LabelEncoder()
X_train[:,1] = labelencoder_X_train.fit_transform(X_train[:,1])
onehotencoder = OneHotEncoder(categories = 'auto')
X_train = onehotencoder.fit_transform(X_train).toarray()



labelencoder_X_test = LabelEncoder()
X_test[:,[1,6]] = labelencoder_X_test.fit_transform(X_test[:,[1,6])
onehotencoder = OneHotEncoder(categorical_features = [0], categories = 'auto')
X_test = onehotencoder.fit_transform(X_test).toarray()

#Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

