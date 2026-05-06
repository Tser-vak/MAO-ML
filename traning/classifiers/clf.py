from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from hyperopt import hp

class ModelFactory:

    @staticmethod
    def get_classifier():
        """Define the Classifier"""
        return {
            'XGBoost': XGBClassifier(eval_metric='logloss',random_state=67,n_jobs=-1,verbosity=0),
            'RandomForest': RandomForestClassifier(random_state=67,n_jobs=-1,verbose=0),
            'LightGBM': LGBMClassifier(random_state=67,n_jobs=-1,verbose=-1)
        }

    @staticmethod
    def get_hyperopt_spaces():
        """Defines the Bayesian search space. Note the 'classifier__' prefix."""
        return {
            "RandomForest": {
                'classifier__n_estimators': hp.quniform('rf_n_estimators', 100, 500, 50),
                'classifier__max_depth': hp.quniform('rf_max_depth', 3, 10, 1),
                'classifier__min_samples_split': hp.quniform('rf_min_samples', 2, 10, 1),
                'classifier__min_samples_leaf': hp.quniform('rf_min_leaf', 1, 5, 1) # Prevents leaves from having just 1 sample
            },
            "XGBoost": {
                'classifier__n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 50),
                'classifier__learning_rate': hp.loguniform('xgb_lr', -5, -1),
                'classifier__max_depth': hp.quniform('xgb_max_depth', 5, 10, 1),
                'classifier__subsample': hp.uniform('xgb_subsample', 0.5, 1.0),
                'classifier__gamma': hp.uniform('xgb_gamma', 0.0, 0.5), # Minimum loss reduction to split
                'classifier__min_child_weight': hp.quniform('xgb_min_child', 1, 10, 1), # Minimum samples in a child node
                'classifier__colsample_bytree': hp.uniform('xgb_colsample', 0.5, 1.0) # Uses fraction of features per tree
            },
            "LightGBM": {
                'classifier__n_estimators': hp.quniform('lgb_n_estimators', 100, 500, 50),
                'classifier__learning_rate': hp.loguniform('lgb_lr', -5, -1),
                'classifier__num_leaves': hp.quniform('lgb_leaves', 20, 100, 5),
                'classifier__max_depth': hp.quniform('lgb_max_depth', 5, 10, 1),
                'classifier__min_child_samples': hp.quniform('lgb_min_child_samples', 5, 30, 1), # Force more samples per leaf
                'classifier__reg_alpha': hp.uniform('lgb_alpha', 0.0, 1.0), # L1 Regularization
                'classifier__reg_lambda': hp.uniform('lgb_lambda', 0.0, 1.0) # L2 Regularization
            }
        }
