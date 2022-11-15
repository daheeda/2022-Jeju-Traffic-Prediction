from os.path import join
import numpy as np
import pandas as pd

from feature.dataset import make_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from category_encoders import *

train_path = join('jeju_data', 'train_new.parquet')
test_path = join('jeju_data', 'test_new.parquet')
holiday_path = join('jeju_data', '국가공휴일.csv')

sample_submission = pd.read_csv('./jeju_data/sample_submission.csv')

x_train, y_train, test = make_dataset(train_path, test_path, holiday_path)
X = x_train.copy()
y = y_train.copy()
df = pd.concat([X, y], axis=1)

import optuna
from optuna import Trial
from optuna.samplers import TPESampler


def objective_xgb(trial: Trial, x, y):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 500, 5000),
        'max_depth': trial.suggest_int('max_depth', 8, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_int('gamma', 1, 3),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 1.0]),
        'random_state': 42
    }

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)

    model = XGBRegressor(**params, tree_method='gpu_hist', gpu_id=0)
    xgb_model = model.fit(x_train, y_train, verbose=False, eval_set=[(x_val, y_val)], early_stopping_rounds=50)
    y_pred = xgb_model.predict(x_val)
    score = mean_absolute_error(y_val, y_pred)

    return score


study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=30)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))


from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


param = study.best_trial.params

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

folds = []

for train_idx, val_idx in skf.split(X, y):
    folds.append((train_idx, val_idx))

XGB_model= {}

for f in range(5):
      print(f'===================================={f+1}============================================')
      train_idx, val_idx = folds[f]
      
      x_train, x_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
      
      XGB = XGBRegressor(**param, tree_method='gpu_hist', gpu_id=0)
      XGB.fit(x_train, y_train)
      
      y_pred = XGB.predict(x_val)
      mae = mean_absolute_error(y_val, y_pred)
      print(f"{f + 1} Fold MAE = {mae}")
      XGB_model[f] = XGB
      print(f'================================================================================\n\n')
              


sample_submission = pd.read_csv('./jeju_data/sample_submission.csv')

for fold in range(5):
    sample_submission['target'] += XGB_model[fold].predict(test)/5    
    

param = study.best_trial.params
sample_submission.to_csv("./submit_xgb_fold.csv", index=False)
df_imp = pd.DataFrame({'imp':XGB.feature_importances_}, index = XGB.feature_names_in_)
df_imp = df_imp[df_imp.imp > 0].sort_values('imp').copy()
print(df_imp)