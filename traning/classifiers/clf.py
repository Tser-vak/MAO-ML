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
                'classifier__max_depth': hp.quniform('rf_max_depth', 3, 30, 1),
                'classifier__min_samples_split': hp.quniform('rf_min_samples', 2, 10, 1)
            },
            "XGBoost": {
                'classifier__n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 50),
                'classifier__learning_rate': hp.loguniform('xgb_lr', -5, -1),
                'classifier__max_depth': hp.quniform('xgb_max_depth', 5, 30, 1),
                'classifier__subsample': hp.uniform('xgb_subsample', 0.5, 1.0)# 0.5 would randomly sample half of the training data 
                                                                              #prior to growing trees. and this will prevent overfitting
            },
            "LightGBM": {
                'classifier__n_estimators': hp.quniform('lgb_n_estimators', 100, 500, 50),
                'classifier__learning_rate': hp.loguniform('lgb_lr', -5, -1),
                'classifier__num_leaves': hp.quniform('lgb_leaves', 20, 100, 5),
                'classifier__max_depth': hp.quniform('lgb_max_depth', 5, 20, 1)
            }
        }
