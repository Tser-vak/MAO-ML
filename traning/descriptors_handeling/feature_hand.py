from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

class FearuteSelector:
    # Feature selector ,LightGBM finding the most important features

    def __init__(self, number_features=25, use_balanced_weights=False):
        self.number_features = number_features
        
        # If we aren't using SMOTE, LightGBM must balance the weights 
        # internally so it doesn't ignore the minority class during RFE.\

        class_weight= "balanced" if use_balanced_weights else None

        self.estimator = RandomForestClassifier(
                                                n_estimators=100,
                                                max_depth=7,
                                                random_state=67,
                                                n_jobs=-1,
                                                verbose=0,
                                                class_weight=class_weight
                                                )

        self.rfe = RFE(estimator=self.estimator,
                      n_features_to_select=self.number_features,
                      step=0.05)
    
    def get_object_rfe(self):
        """Returns the un-fitted RFE object for the pipeline."""
        return self.rfe