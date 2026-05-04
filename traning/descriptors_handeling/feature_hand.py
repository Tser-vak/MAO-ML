from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier

class FearuteSelector:
    # Feature selector ,LightGBM finding the most important features

    def __init__(self, number_features=40, use_balanced_weights=False):
        self.number_features = number_features
        
        # If we aren't using SMOTE, LightGBM must balance the weights 
        # internally so it doesn't ignore the minority class during RFE.\

        class_weight= "balanced" if use_balanced_weights else None

        self.estimator = LGBMClassifier(random_state=42,
                                        n_jobs=-1,
                                        verbose=-1,
                                        class_weight=class_weight)

        self.rfe = RFE(estimator=self.estimator,
                      n_features_to_select=self.number_features,
                      step=0.05)
    
    def get_object_rfe(self):
        """Returns the un-fitted RFE object for the pipeline."""
        return self.rfe