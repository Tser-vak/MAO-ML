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
        """Defines the Bayesian search space with aggressive regularization to prevent overfitting."""
        return {
            "RandomForest": {
                'classifier__n_estimators': hp.quniform('rf_n_estimators', 100, 500, 50),
                'classifier__max_depth': hp.quniform('rf_max_depth', 3, 7, 1),
                'classifier__min_samples_split': hp.quniform('rf_min_samples_split', 5, 20, 1), 
                'classifier__min_samples_leaf': hp.quniform('rf_min_samples_leaf', 5, 15, 1) 
            },
            "XGBoost": {
                'classifier__n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 50),
                'classifier__learning_rate': hp.loguniform('xgb_learning_rate', -5, -1),
                'classifier__max_depth': hp.quniform('xgb_max_depth', 3, 7, 1),
                'classifier__subsample': hp.uniform('xgb_subsample', 0.5, 0.9), 
                'classifier__gamma': hp.uniform('xgb_gamma', 0.5, 5.0), 
                'classifier__min_child_weight': hp.quniform('xgb_min_child_weight', 5, 15, 1), 
                'classifier__colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.5, 0.9),
                'classifier__reg_alpha': hp.uniform('xgb_reg_alpha', 0.0, 2.0), 
                'classifier__reg_lambda': hp.uniform('xgb_reg_lambda', 1.0, 5.0)  
            },
            "LightGBM": {
                'classifier__n_estimators': hp.quniform('lgb_n_estimators', 100, 500, 50),
                'classifier__learning_rate': hp.loguniform('lgb_learning_rate', -5, -1),
                'classifier__num_leaves': hp.quniform('lgb_num_leaves', 10, 31, 1), 
                'classifier__max_depth': hp.quniform('lgb_max_depth', 3, 7, 1), 
                'classifier__min_child_samples': hp.quniform('lgb_min_child_samples', 15, 40, 1), 
                'classifier__reg_alpha': hp.uniform('lgb_reg_alpha', 0.0, 2.0), 
                'classifier__reg_lambda': hp.uniform('lgb_reg_lambda', 0.0, 3.0) 
            }
        }
