# -*- coding: utf-8 -*-
'''
CASE1: Bike Sharing Demand (Kaggle)
Please download the data files from Kaggle
https://www.kaggle.com/c/bike-sharing-demand/overview
'''
# import packages
import pandas as pd
from sklearn.linear_model import LinearRegression

# load data
train = pd.read_csv('./Desktop/train.csv')
test = pd.read_csv('./Desktop/test.csv')

# pick the columns for making submission file
label = train['count']
dt = test['datetime']

# select the features
features = ['holiday', 'workingday', 'atemp', 'humidity', 'windspeed']
train = train[features]
test = test[features]

# building linear regression model and setting model's parameter
lr = LinearRegression(n_jobs=-1)

# fit model
lr.fit(train, label)

# make predictions
pred = lr.predict(test)

# make a subimission file
submission = pd.DataFrame({
        'datetime': dt, 
        'count': [max(0, y) for y in pred]})
submission.to_csv('/Users/kai/Desktop/submit.csv', index=False)
