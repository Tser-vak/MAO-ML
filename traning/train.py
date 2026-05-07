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
    precision_score, recall_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load OPP functions
from data.data import DataProcessor, DataConv, ModelExporter # Data handeling processes
from classifiers.clf import ModelFactory # Budle of Classifier and hyper-parameter tuning 
from descriptors_handeling.feature_hand import FearuteSelector  # Feature selection (RFE) object
from visualization.visual import ModelVisualizer # Budle of Visualizations tools

def format_params(params):
    """Casts hyperopt float parameters to integers where needed."""
    formatted = params.copy()
    for key in formatted.keys():
        if any(kw in key for kw in ['depth', 'estimators', 'leaves', 'samples']):
            formatted[key] = int(formatted[key])
    return formatted

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
    rfe_smote = FearuteSelector(number_features=25,use_balanced_weights=False)
    rfe_no_smote = FearuteSelector(number_features=25,use_balanced_weights=True)

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

    print('Phase 2 : Hyperparameter Tunning\n')
    for model_name , base_model in models.items():
        for use_smote in [True,False]:
            path_name = 'SMOTE' if use_smote else 'No_SMOTE'
            run_name = f"{model_name}_{path_name}"
            print(f"Optimizing {run_name}...")

            if not use_smote:
                if model_name in ['RandomForest','LightGBM']:
                    base_model.set_params(class_weight='balanced')

                elif model_name == 'XGBoost':
                    # XGBoost has no built-in class_weight parameter, so we use scale_pos_weight.
                    # The formula is (number of negative samples) / (number of positive samples)
                    count = pd.Series(y_train).value_counts()
                    weights = count.min() / count.max()
                    base_model.set_params(scale_pos_weight= weights)
            else:
                if model_name in ['RandomForest','LightGBM']:
                    base_model.set_params(class_weight=None)
                elif model_name == 'XGBoost':
                    base_model.set_params(scale_pos_weight=None)

            run_counter = 1
            def objective(params):
                nonlocal run_counter
                # Format parameters so n_estimators, max_depth, etc are integers
                params = format_params(params)
                
                # Create pipeline
                if use_smote:
                    pipeline = ImbPipeline([('scaler', scaler) , ('rfe',rfe_smote.get_object_rfe()) , ('smote',smote )  , ('classifier',base_model)])
                else:
                    pipeline = SklearnPipeline([('scaler', scaler), ('rfe',rfe_no_smote.get_object_rfe()), ('classifier',base_model)])  
                
                #Set parameters
                pipeline.set_params(**params)
                
                # Cross validation list 
                score = ['matthews_corrcoef','roc_auc','balanced_accuracy','precision','recall']
                
                # Calculate Cross-validation Scores
                cv_results = cross_validate(
                    estimator=pipeline, X=X_train, y=y_train,
                    cv=cv_strategy, scoring=score, n_jobs=1,
                    return_train_score=True)
                
                # Mean values 
                mcc_mean = np.mean(cv_results['test_matthews_corrcoef'])
                mcc_mean_train = np.mean(cv_results['train_matthews_corrcoef'])
                roc_mean = np.mean(cv_results['test_roc_auc'])
                ba_mean = np.mean(cv_results['test_balanced_accuracy'])
                prec_mean = np.mean(cv_results['test_precision'])
                rec_mean = np.mean(cv_results['test_recall'])

                # ===================================Penalize LOSS Calculation Due to script selecting Good  Mcc_test with ===================================
                # =================================== Bad train Mcc (Overfitting) ===================================
                alpha = 0.4 # penalty weight
                mcc_gap = abs(mcc_mean - mcc_mean_train) # Gap between train and test MCC
                penalty_loss = -mcc_mean + (alpha * mcc_gap) # Penalized Loss
                # ============================================================================================================================================

                with mlflow.start_run(run_name=f"Run {run_counter}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        'Mean_mcc_test_cv': mcc_mean,
                        'Mean_mcc_train_cv': mcc_mean_train,
                        'Mean_roc_auc_test_cv': roc_mean,
                        'Mean_balanced_accuracy_test_cv': ba_mean,
                        'Mean_precision_test_cv': prec_mean,
                        'Mean_recall_test_cv': rec_mean,
                        'Custom_Penalized_Loss': penalty_loss
                    })

                run_counter += 1
                return {'loss': round(penalty_loss, 3), 'status': STATUS_OK}
            
            with mlflow.start_run(run_name = run_name):
                # Initialize hyper-parameter tuning
                trials = Trials()
                raw_best_param = fmin(
                    fn=objective,
                    space=spaces[model_name],
                    algo=tpe.suggest,
                    max_evals=35,
                    trials=trials,
                    rstate=np.random.default_rng(67)
                )

                best_params = format_params({ f'classifier__{k.split('_',1)[1]}': v for k,v in raw_best_param.items()})

                best_mcc_this_run = -trials.best_trial['result']['loss']
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_mcc", best_mcc_this_run)
                print(f"Best CV MCC for {run_name}: {best_mcc_this_run:.4f}")

                if best_mcc_this_run > global_best_mcc:
                    global_best_mcc = best_mcc_this_run
                    global_best_config = {
                        'model_name': model_name,
                        'base_model': base_model,
                        'use_smote': use_smote,
                        'params': best_params
                    }
    # ===================================================================== 
    # Phase 3: Final Evaluation of model and visualization
    # ===================================================================== 
    print(f"\n🏆 Phase 3: GLOBAL WINNER -> {global_best_config['model_name']} ({'SMOTE' if global_best_config['use_smote'] else 'No-SMOTE'})")
    print("Training final model on full 80% train set...")
    
    if global_best_config['use_smote']:
        final_pipeline = ImbPipeline([('scaler', scaler), ('smote', smote), ('rfe', rfe_smote.get_object_rfe()), ('classifier', global_best_config['base_model'])])
    else:
        final_pipeline = SklearnPipeline([('scaler', scaler), ('rfe', rfe_no_smote.get_object_rfe()), ('classifier', global_best_config['base_model'])])
        
    final_pipeline.set_params(**global_best_config['params'])
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    
    test_mcc = matthews_corrcoef(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_proba)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_metrics({"holdout_mcc": test_mcc,
                         "holdout_roc_auc": test_roc,
                         "holdout_bal_acc": test_bal_acc,
                         "holdout_precision": test_prec,
                         "holdout_recall": test_rec
                         })

    print("\n" + "="*45)
    print("   FINAL HOLDOUT RESULTS ")
    print("="*45)
    print(f"MCC Score:         {test_mcc:.4f}  <-- MAIN METRIC")
    print(f"ROC-AUC:           {test_roc:.4f}")
    print(f"Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"Precision:         {test_prec:.4f}")
    print(f"Recall:            {test_rec:.4f}")
    print("-" * 45)
    print("Confusion Matrix Breakdown:")
    print(f"True Negatives  (Correct Inactives): {cm[0][0]}")
    print(f"False Positives (Wasted Lab Time):   {cm[0][1]}")
    print(f"False Negatives (Missed Actives):    {cm[1][0]}")
    print(f"True Positives  (Correct Actives):   {cm[1][1]}")
    print("="*45 + "\n")

    print("Generating visual plots...")
    visualizer = ModelVisualizer(global_best_config['model_name'], global_best_config['use_smote'])
    visualizer.log_confusion_matrix(cm)
    visualizer.log_roc_curve(y_test, y_proba)

    # ===================================================================== 
    # PHASE 4: ONNX EXPORT 
    # ===================================================================== 
    print("\nPhase 4: Exporting clean production ONNX model...") 
    
    # Grab the fitted RFE mask from the winning pipeline
    fitted_rfe = final_pipeline.named_steps['rfe'] 
    support_mask = fitted_rfe.support_
    
    # Map the mask back to the original training dataframe columns
    selected_features = X_train.columns[support_mask].tolist() 
    
    # Let the Exporter handle the slicing and ONNX conversion
    ModelExporter.export_production_pipeline(
        final_pipeline=final_pipeline,
        support_mask=support_mask,
        selected_features=selected_features,
        model_name=global_best_config['model_name']
    )

if __name__ == "__main__" : 
    main()
