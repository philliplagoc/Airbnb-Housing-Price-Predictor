#!/usr/bin/env python
# coding: utf-8

# Import packages
import pandas as pd
import numpy as np
import pickle
import os
import time

import lightgbm as lgb
from xgboost.sklearn import XGBRegressor

from scipy import stats
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost.sklearn import XGBRegressor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Get data
df = pd.read_csv("cleaned_listings.csv")
df = df.drop(columns = ["Unnamed: 0"])

X = df[ [var for var in df.columns if var != "log_price"] ].values
y = df['log_price'].values

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



# Necessary Functions
"""
Gets the saved model if it exists as a pickle.dat file.

Returns the best model after loading it in via pickle.

Params:
model_name - The name of the model.
saved_model_path - Relative path name for where the pickle.dat file lives.
"""
def get_saved_model(model_name, saved_model_path):
    # Load in saved model
    best_model = pickle.load(open(saved_model_path, 'rb'))

    # Compute saved model's RMSE
    best_model_preds = best_model.predict(X_test)
    print(model_name, 'MAE: %.4f' % mean_absolute_error(y_test, best_model_preds))

    return best_model

"""
Uses RandomizedSearch to find the best parameters for the model.

Returns the best parameters afterwards.

Parameters:
model_name - The name of the model.
X_train_val - The train and validation features.
y_train_val - The training and validation responses.
rs_clf - The RandomizedSearch model.
"""
def get_best_params(model_name, X_train_val, y_train_val, rs_clf):
    print('Starting to train', model_name, '...')

    start = time.time()
    rs_clf.fit(X_train_val, y_train_val)
    print("RandomizedSearch took %.2f seconds to complete." % (time.time() - start))

    return rs_clf.best_params_

"""
Fits the given model and saves the model as a pickle.dat file.

Returns the MAE of the best model.

Parameters:
model - The model to fit.
saved_model_path - Relative path name for where the pickle.dat file lives.
X_train_val - The train and validation features.
y_train_val - The training and validation responses.
X_test - The testing features.
y_test - The testing responses.
"""
def fit_best_model(model, saved_model_path, X_train_val, X_test, y_train_val, y_test):
    model.fit(X_train_val, y_train_val)
    pickle.dump(model, open(saved_model_path, 'wb'))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print("MAE: %.4f" % mae)    
    return mae



# Necessary Variables
n_iters = 50
n_folds = 5
n_estimators = np.arange(10, 100, 10)
early_stopping_rounds = 15
eval_set = (X_test, y_test)
eval_metric = 'mae'


### XGB
xgb_fit_dict = {
    'eval_metric': eval_metric,
    "early_stopping_rounds": early_stopping_rounds,
    "eval_set": [eval_set],
    'verbose': 100
}

xgb_param_dict = {
    'n_estimators': n_estimators,
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'learning_rate': [0.05, 0.1, 0.3],
    'subsample': stats.uniform(loc=0.2, scale=0.8),
    'colsample_bytree': stats.uniform(loc=0.4, scale=0.6),
    'gamma': [0.0, 0.1, 0.2],
    'max_depth': [5, 7, 10],
    'min_child_samples': stats.randint(100, 500), 
    "objective": ["reg:squarederror"],
    'alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}

xgb_model = XGBRegressor(seed = 42, metric = 'None', n_jobs = 4, silent = True)

rs_clf = RandomizedSearchCV(xgb_model, random_state = 42, 
                            param_distributions = xgb_param_dict, 
                            n_iter = n_iters, 
                            cv = n_folds, 
                            scoring = 'neg_mean_absolute_error', 
                            verbose = False) 

xgb_model = XGBRegressor(seed = 42, metric = 'None', n_jobs = 4, silent = True)

rs_clf = RandomizedSearchCV(xgb_model, random_state = 42, 
                            param_distributions = xgb_param_dict, 
                            n_iter = n_iters, 
                            cv = n_folds, 
                            scoring = 'neg_mean_absolute_error', 
                            verbose = False) 

xgb_saved_model_path = "models/xgb_model.pickle.dat"

if os.path.exists(xgb_saved_model_path):
    # Load in existing model
    xgb_best_model = get_saved_model("XGB", xgb_saved_model_path)
else:
    # Train a new model
    xgb_best_model_params = get_best_params("XGB", X_train_val, y_train_val, rs_clf)
    
    # Train using best parameters
    xgb_best_model = XGBRegressor(**xgb_best_model_params, random_state = 42)
    
    # Fit best model
    xgb_best_model.fit(X_train_val, y_train_val)
    results = fit_best_model(xgb_best_model, xgb_saved_model_path, X_train_val, X_test, y_train_val, y_test)
    


### LGBM
lgb_fit_dict = {
    'eval_metric': eval_metric,
    "early_stopping_rounds": early_stopping_rounds,
    "eval_set": [eval_set],
    'verbose': 100
}

lgb_param_dict = {
    'n_estimators': n_estimators,
    'num_leaves': stats.randint(6, 50), 
    'learning_rate': [0.05, 0.1, 0.3],
    'min_child_samples': stats.randint(100, 500), 
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': stats.uniform(loc=0.2, scale=0.8), 
    'colsample_bytree': stats.uniform(loc=0.4, scale=0.6),
    'alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}

lgb_model = lgb.LGBMRegressor(seed = 42, metric = 'None', n_jobs = 4, silent = True)

rs_clf = RandomizedSearchCV(lgb_model, random_state = 42, param_distributions = lgb_param_dict, 
                            n_iter = n_iters, cv = n_folds, 
                            scoring = 'neg_mean_absolute_error', 
                            verbose = False) 

lgb_saved_model_path = "models/lgb_model.pickle.dat"

if os.path.exists(lgb_saved_model_path):
    # Load in existing model
    lgb_best_model = get_saved_model("LGBM", lgb_saved_model_path)
else:
    # Train a new model
    lgb_best_model_params = get_best_params("LGBM", X_train_val, y_train_val, rs_clf)
    
    # Train using best parameters
    lgb_best_model = lgb.LGBMRegressor(**lgb_best_model_params, random_state = 42)
    
    # Fit best model
    lgb_best_model.fit(X_train_val, y_train_val)
    results = fit_best_model(lgb_best_model, lgb_saved_model_path, X_train_val, X_test, y_train_val, y_test)
    



### GradientBoost
gb_param_dict = {
    'n_estimators': n_estimators,
    'max_depth': [5, 7, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'loss': ['huber', 'lad'],
    'learning_rate': [0.05, 0.1, 0.3],
    'subsample': stats.uniform(loc=0.2, scale=0.8),
    'min_samples_leaf': stats.randint(100, 500)
}

gb_model = GradientBoostingRegressor(random_state = 42, n_iter_no_change = early_stopping_rounds, tol = 0.1)

rs_clf = RandomizedSearchCV(gb_model, random_state = 42, 
                            param_distributions = gb_param_dict, 
                            n_iter = n_iters, 
                            cv = n_folds, 
                            scoring = 'neg_mean_absolute_error', 
                            verbose = False) 

gb_saved_model_path = "models/gb_model.pickle.dat"

if os.path.exists(gb_saved_model_path):
    # Load in existing model
    gb_best_model = get_saved_model("GradientBoost", gb_saved_model_path)
else:
    # Train a new model
    gb_best_model_params = get_best_params("GradientBoost", X_train_val, y_train_val, rs_clf)
    
    # Train using best parameters
    gb_best_model = GradientBoostingRegressor(**gb_best_model_params, random_state = 42)
    
    # Fit best model
    gb_best_model.fit(X_train_val, y_train_val)
    results = fit_best_model(gb_best_model, gb_saved_model_path, X_train_val, X_test, y_train_val, y_test)



### Adaboost
ab_param_dict = {
    'learning_rate': [0.05, 0.1, 0.3],
    'loss': ['linear', 'square', 'exponential'],
    'n_estimators': n_estimators
}

ab_model = AdaBoostRegressor(random_state = 42)

rs_clf = RandomizedSearchCV(ab_model, param_distributions = ab_param_dict, random_state = 42,
                            n_iter = n_iters, cv = n_folds, 
                            scoring = 'neg_mean_absolute_error', 
                            verbose = True) 

ab_saved_model_path = "models/ab_model.pickle.dat"

if os.path.exists(ab_saved_model_path):
    # Load in existing model
    ab_best_model = get_saved_model("AdaBoost", ab_saved_model_path)
else:
    # Train a new model
    ab_best_model_params = get_best_params("AdaBoost", X_train_val, y_train_val, rs_clf)
    
    # Train using best parameters
    ab_best_model = AdaBoostRegressor(**ab_best_model_params, random_state = 42)
    
    # Fit best model
    ab_best_model.fit(X_train_val, y_train_val)
    results = fit_best_model(ab_best_model, ab_saved_model_path, X_train_val, X_test, y_train_val, y_test)



# Bagging models
### Aggregate average score together
best_xgb_preds = xgb_best_model.predict(X_test)
best_lgb_preds = lgb_best_model.predict(X_test)
best_gb_preds = gb_best_model.predict(X_test)
best_ab_preds = ab_best_model.predict(X_test)

bagged_preds = (best_xgb_preds + best_lgb_preds + best_ab_preds + best_gb_preds) / 4

print("Bagging MAE:", mean_absolute_error(y_test, bagged_preds))

