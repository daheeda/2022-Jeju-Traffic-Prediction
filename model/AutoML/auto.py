#AutoGluon 쓸때는 numpy 1.23으로
#pycaret 쓸때는 numpy 1.20으로

import os
import torch
import random
import numpy as np
import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, train_test_split


train_x = pd.read_csv('./EngineeringV4_train_x_1.csv')
train_y = pd.read_csv('./EngineeringV4_train_y_1.csv')
test_x = pd.read_csv('./EngineeringV4_test_x_1.csv')


#######################Modeling########################


train = pd.concat([train_x,train_y], axis = 1)



train_data = TabularDataset(train)
test_data = TabularDataset(test_x)

#train_data = TabularDataset('./train_data.csv')
#test_data = TabularDataset('./test_data.csv')


print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")

#save_path = 'Models-predict'  # specifies folder to store trained models

predictor = TabularPredictor(label='target',  eval_metric='mean_absolute_error').fit(train_data, presets='high_quality',  ag_args_fit={'num_gpus': 1})


print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")


#predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data)

print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")

y_pred_final = pd.DataFrame(y_pred, columns=['target'])

#######################submission########################
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['target'] = y_pred_final
sample_submission.to_csv("./submit5_auto.csv", index = False)


print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")


