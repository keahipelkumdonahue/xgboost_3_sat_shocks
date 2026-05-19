import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt.space import Real, Integer
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

type = 'sunward'  # change this line for sunward vs earthward model training

######################################### DATA AND TRAINING #########################################

# Load and prepare data
training_data = joblib.load('training_data/training_data_{}.pkl'.format(type))
# columns=['t_a', 'lag_ab', 'lag_ac',
#                               'pos_a_x', 'pos_a_y', 'pos_a_z',
#                               'pos_b_x', 'pos_b_y', 'pos_b_z',
#                               'pos_c_x', 'pos_c_y', 'pos_c_z',
#                               'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
#                               'v_true_x', 'v_true_y', 'v_true_z',
#                               'v_calc_x', 'v_calc_y', 'v_calc_z'])

# Train model
train = training_data.sample(frac=0.8, random_state=42)
test = training_data.drop(train.index)

X_train = train[['orb_norm_x', 'orb_norm_y', 'orb_norm_z',
                 'v_calc_x', 'v_calc_y', 'v_calc_z']]
y_train = train[['v_true_x', 'v_true_y', 'v_true_z']]

X_test = test[['orb_norm_x', 'orb_norm_y', 'orb_norm_z',
               'v_calc_x', 'v_calc_y', 'v_calc_z']]
X_additional = test[['lag_ab', 'lag_ac',
                 'pos_a_x', 'pos_a_y', 'pos_a_z',
                 'pos_b_x', 'pos_b_y', 'pos_b_z',
                 'pos_c_x', 'pos_c_y', 'pos_c_z']]
y_test = test[['v_true_x', 'v_true_y', 'v_true_z']]

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

# Define the search space for hyperparameters
search_spaces = {
    'estimator__n_estimators': Integer(50, 300),
    'estimator__max_depth': Integer(3, 10),
    'estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'estimator__min_child_weight': Integer(1, 7),
    'estimator__subsample': Real(0.5, 1.0),
    'estimator__colsample_bytree': Real(0.5, 1.0)
}

# Create the base model
base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
multi_model = MultiOutputRegressor(base_model)

# Create BayesSearchCV object
bayes_search = BayesSearchCV(
    estimator=multi_model,
    search_spaces=search_spaces,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=5,       # Cross-validation folds
    n_jobs=-1,  # Use all CPU cores
    random_state=42,
    scoring='neg_mean_squared_error'
)

'''
# Test grid search
grid = {'estimator__n_estimators': [50, 100, 200, 300],
        'estimator__max_depth': [3, 5, 7, 10],
        'estimator__learning_rate': [0.01, 0.1, 0.3]}
bayes_search = GridSearchCV(estimator=multi_model, param_grid=grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
'''

'''
# Test random search
from scipy.stats import uniform, randint
param_distributions = {
    'estimator__n_estimators': randint(50, 300),
    'estimator__max_depth': randint(3, 10),
    'estimator__learning_rate': uniform(0.01, 0.3)}
bayes_search = RandomizedSearchCV(estimator=multi_model, param_distributions=param_distributions, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
'''

# Perform hyperparameter optimization
print("Running Bayesian optimization")
bayes_search.fit(X_train_np, y_train_np)

######################################### TEST SET AND SAVE #########################################

best_model = bayes_search.best_estimator_

# Print the best parameters and score
print("\nBest parameters found:")
print(bayes_search.best_params_)
print(f"\nBest cross-validation score: {-bayes_search.best_score_:.6f} MSE")

#### Use for pulling test statistics w/o re-training
best_model = joblib.load('{}_xgb_model.pkl'.format(type))
X_test = joblib.load('training_data/test_data/{}_x.pkl'.format(type))
y_test = joblib.load('training_data/test_data/{}_y.pkl'.format(type))

# Use the best model for predictions
X_test_np = X_test.to_numpy()
y_pred = best_model.predict(X_test_np)
y_test_np = y_test.to_numpy()

mse_best = mean_squared_error(y_test_np, y_pred, multioutput='raw_values')
r2_best = r2_score(y_test_np, y_pred, multioutput='raw_values')

print("\nBest Model Performance:")
print(f"MSE for each output: {mse_best}")
print(f"R² for each output: {r2_best}")

# Save the best model
joblib.dump(best_model, '{}_xgb_model.pkl'.format(type))
joblib.dump(X_test, 'training_data/test_data/{}_x.pkl'.format(type))
joblib.dump(y_test, 'training_data/test_data/{}_y.pkl'.format(type))
joblib.dump(y_pred, 'training_data/test_data/{}_y_pred.pkl'.format(type))

