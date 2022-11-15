from os.path import join
import pandas as pd
from feature.dataset import make_dataset

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error


train_path = join('jeju_data', 'train_new.parquet')
test_path = join('jeju_data', 'test_new.parquet')
holiday_path = join('jeju_data', '국가공휴일.csv')
tour_path = join('jeju_data', 'AT4_2000.csv')
submission_path = join('jeju_data', 'sample_submission.csv')


x_train, y_train, test = make_dataset(train_path, test_path, holiday_path, tour_path)
sample_submission = pd.read_csv(submission_path)


X = x_train.copy()
y = y_train.copy()


params = {'n_estimators': 4693, 'max_depth': 15, 'min_child_weight': 4, 'gamma': 1, 'learning_rate': 0.018,
          'colsample_bytree': 0.9015018647603987, 'lambda': 1.386941960727803, 'alpha': 0.10534800837686535, 'subsample': 1.0}

skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=404)

folds = []

for train_idx, val_idx in skf.split(X, y):
    folds.append((train_idx, val_idx))

XGB_model = {}

for f in range(9):
    print(
        f'===================================={f+1}============================================')
    train_idx, val_idx = folds[f]

    x_train, x_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]

    XGB = XGBRegressor(**params, tree_method='gpu_hist',
                       gpu_id=0, random_state=404)
    XGB.fit(x_train, y_train)

    y_pred = XGB.predict(x_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"{f + 1} Fold MAE = {mae}")
    XGB_model[f] = XGB
    print(f'================================================================================\n\n')


for fold in range(9):
    sample_submission['target'] += XGB_model[fold].predict(test)/9
sample_submission.to_csv("./submit_xgb_9fold.csv", index=False)
