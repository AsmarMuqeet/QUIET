import optuna
import joblib
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from optuna.integration import LightGBMPruningCallback
import numpy as np

#import util as U
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb

def objective(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X.values, y.values)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(objective="regression", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(1000),
                LightGBMPruningCallback(trial, "l2")
            ],  # Add a pruning callback
        )
        preds = model.predict(X_test, num_iteration=model.best_iteration_)
        cv_scores[idx] = mean_squared_error(y_test, preds)

    return np.mean(cv_scores)

if __name__ == '__main__':
    #U.data_generation()
    df = pd.read_csv("data/training.csv")
    df[df["ideal_prob"]==-1] = 0
    y = df["ideal_prob"]
    df.drop(["ideal_prob"], axis=1, inplace=True)

    # study = optuna.create_study(direction="minimize", study_name="LGBM Regressor")
    # func = lambda trial: objective(trial, df, y)
    # study.optimize(func, n_trials=20)
    #
    # print(f"\tBest value (mse): {study.best_value:.5f}")
    # print(f"\tBest params:")
    #
    # for key, value in study.best_params.items():
    #     print(f"\t\t{key}: {value}")

    X_train, X_test, y_train, y_test = train_test_split(df,y, test_size=0.2, random_state=42)
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'n_estimators': 10000,
        'learning_rate': 0.09,
        'num_leaves': 240,
        'max_depth': 12,
        'min_data_in_leaf': 400,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'feature_fraction': 0.3,
        "num_iterations": 100000
    }
    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(1000)])

    y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration_)
    print('The mse of prediction is:', round(mean_squared_error(y_pred, y_train), 5))
    joblib.dump(gbm, 'model/ML.model')
    print('Model saved to model/ML.model')