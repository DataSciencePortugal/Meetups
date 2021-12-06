import argparse
import warnings
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.xgboost


def train(space):
    # Load Iris dataset from scikit-learn and configure XGBoost data matrices
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        df_iris, iris.target, random_state=0)

    # Set autologging before constructing the dataset to infer signature
    mlflow.xgboost.autolog()

    # Configure training data matrices
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    with mlflow.start_run(run_name=f'timestamp-{datetime.now()}'):
        # specify parameters via map
        param = {
            'max_depth': int(space['max_depth']),
            'gamma': space['gamma'],
            'reg_alpha': space['reg_alpha'],
            'reg_lambda': space['reg_lambda'],
            'colsample_bytree': space['colsample_bytree'],
            'min_child_weight': space['min_child_weight'],
            'eta': 0.3,  # the training step for each iteration
            'verbosity': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'eval_metric': 'mlogloss',
            'num_class': pd.Series(iris.target).nunique()
        }
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)

        # make prediction
        preds = bst.predict(dtest)
        best_preds = np.asarray([np.argmax(line) for line in preds])
        f1 = f1_score(best_preds, y_test, average='macro')
        mlflow.log_metric('f1_score', f1)

        # Plot confusion matrix and log artifact
        cm = confusion_matrix(y_test, best_preds, labels=pd.Series(iris.target).unique())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pd.Series(iris.target).unique())
        mlflow.log_figure(disp.plot().figure_, 'cm.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost train on Iris.')
    parser.add_argument('max_depth', type=float)
    parser.add_argument('gamma', type=float)
    parser.add_argument('reg_alpha', type=float)
    parser.add_argument('reg_lambda', type=float)
    parser.add_argument('colsample_bytree', type=float)
    parser.add_argument('min_child_weight', type=float)
    parser.add_argument('eta', type=float)
    parser.add_argument('objective', type=str)
    parser.add_argument('eval_metric', type=str)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    train(vars(args))
