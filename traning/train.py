import mlflow
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score, 
    precision_score, recall_score, log_loss, confusion_matrix
)

# Load OPP functions
from data.data import DataProcessor,DataConv # Data handeling processes
from classifiers.clf import ModelFactory # Budle of Classifier and hyper-parameter tuning 
from descriptors_handeling.feature_hand import FearuteSelector  # Feature selection (RFE) object
from visualization.visual import ModelVisualizer # Budle of Visualizations tools

def main():
    print('Phase 1 Laoding and Cleaning Data\n')
    file = r'C:\Users\Max\Desktop\MAO-ML\traning\data\MAOB_PD_Activity_with_descriptors.csv'
    cleaner = DataProcessor(file_path=file,ignore_desc = ["Molecule ChEMBL ID","Smiles","Standard Type","Standard Relation","Standard Value","Standard Units","Label"],
         labels = "Label")
    # Load and Clean Data (Info data.py)
    cleaner.load_data()

    # Split Data
    (X_train, X_test, y_train, y_test) = cleaner.get_split()
    
    # Initialize scaler and smote
    scaler = DataConv.Scaler('standard')
    smote = DataConv.get_smote()
    
    # Initialize feature selection (RFE)
    rfe_smote = FearuteSelector(number_features=40,use_balanced_weights=False)
    rfe_no_smote = FearuteSelector(number_features=40,use_balanced_weights=True)

    # Initialize Cross Validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize Models and Hyper-parameter tuning spaces
    models = ModelFactory.get_classifier()
    spaces = ModelFactory.get_hyperopt_spaces()

    # Start instance of mlflow
    mlflow.set_experiment('MAO_ML_Results')

    # Set parameters for tracking
    global_best_mcc = -1.0
    global_config = {}

    # Phase 2 : 


if __name__ == "__main__" : 
    main()
