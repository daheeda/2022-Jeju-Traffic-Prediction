from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

seed = 42
train = pd.read_parquet('./jeju_data/train_48.parquet')
test = pd.read_parquet('./jeju_data/test_48.parquet')
X = train.drop(['target'], axis=1)
y = train['target']

xgb_param = {'n_estimators': 4693, 'max_depth': 15, 'min_child_weight': 4, 'gamma': 1, 'learning_rate': 0.018,
             'colsample_bytree': 0.9015018647603987, 'lambda': 1.386941960727803, 'alpha': 0.10534800837686535, 'subsample': 1.0}

cat_param = {'iterations': 589, 'learning_rate': 0.33865268174378255, 'depth': 16, 'min_data_in_leaf': 30,
             'reg_lambda': 61.46317610780882, 'subsample': 0.7550764410597048,
             'random_strength': 48.76480960330878, 'od_wait': 62, 'leaf_estimation_iterations': 6,
             'bagging_temperature': 1.2692205293143553, 'colsample_bylevel': 0.14863052892541012}

lgb_param = {'max_depth': 8, 'learning_rate': 0.005061576888752304, 'n_estimators': 3928, 'min_child_samples': 62, 'subsample': 0.46147273880126555, 'lambda_l1': 2.5348407664333426e-07,
             'lambda_l2': 3.3323645788192616e-08, 'num_leaves': 222, 'feature_fraction': 0.7606690070459252, 'bagging_fraction': 0.8248435466776274, 'bagging_freq': 1}

hgb_param = {'loss': 'absolute_error', 'learning_rate': 0.0249816047538945, 'max_iter': 2877,
             'max_leaf_nodes': 189, 'max_depth': 28, 'min_samples_leaf': 104, 'l2_regularization': 0.05, 'random_state': 42}


def get_stacking_model(model, X, y, test, n_fold):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    folds = []

    for train_idx, val_idx in skf.split(X, y):
        folds.append((train_idx, val_idx))

    fold_model= {}

    for f in range(n_fold):
        print(f'===================================={f+1}============================================')
        train_idx, val_idx = folds[f]
        
        x_train, x_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"{f + 1} Fold MAE = {mae}")
        fold_model[f] = model
        print(f'================================================================================\n\n')
                
    sample_submission_train = pd.read_parquet('./jeju_data/train_id_target.parquet')
    sample_submission_test = pd.read_csv('./jeju_data/sample_submission.csv')

    for fold in range(n_fold):
        sample_submission_train['target'] += fold_model[fold].predict(X)/n_fold
        sample_submission_test['target'] += fold_model[fold].predict(test)/n_fold
    
    return sample_submission_train['target'], sample_submission_test['target']


XGB = XGBRegressor(**xgb_param)
Cat = CatBoostRegressor(**cat_param)
lgbm = LGBMRegressor(**lgb_param)
hgr = HistGradientBoostingRegressor(**hgb_param)


xgb_train_pred, xgb_test_pred = get_stacking_model(XGB, X, y, test, 7)
Cat_train_pred, Cat_test_pred = get_stacking_model(Cat, X, y, test, 7)
lgbm_train_pred, lgbm_test_pred = get_stacking_model(lgbm, X, y, test, 7)
hgr_train_pred, hgr_test_pred = get_stacking_model(hgr, X, y, test, 7)

Stack_final_X_train = pd.concat([xgb_train_pred, Cat_train_pred, lgbm_train_pred, hgr_train_pred], axis=1)
Stack_final_X_test = pd.concat([xgb_test_pred, Cat_test_pred, lgbm_test_pred, hgr_test_pred], axis=1)

meta_model = XGBRegressor()
meta_model.fit(Stack_final_X_train, y)
stack_final = meta_model.predict(Stack_final_X_test)


sample_submission = pd.read_csv('./jeju_data/sample_submission.csv')
sample_submission['target'] = stack_final
sample_submission.to_csv("stacking.csv", index=False)

