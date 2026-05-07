import pandas as pd
import numpy as np
import json 
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class DataProcessor:
    """
    Handles the full data loading, cleaning, and splitting pipeline.
    
    Args:
        file_path (str): Path to the CSV dataset.
        ignore_desc (list): Column names to drop before processing (e.g. metadata columns).
        labels (str): Name of the target/label column.
    """

    def __init__(self, file_path, ignore_desc=None, labels=None):
        self.file_path = file_path
        # Default to empty list if no columns are specified to drop
        self.ignore_desc = ignore_desc if ignore_desc is not None else []

        if labels is not None:
            self.labels = labels
        else:
            raise ValueError("WARNING: The label column is missing")
        
        self.lb_data = None  # Will hold the raw label Series after load
        self.data = None     # Will hold the feature DataFrame after load

        # Explicit mapping: 0 = Not Active (negative class), 1 = Active (positive class).
        # Using a manual map instead of LabelEncoder to guarantee consistent ordering.
        self.label_map = {'Not Active': 0, 'Active': 1}

    def load_data(self):
        """
        Loads the CSV, removes unwanted columns, runs diagnostics,
        imputes missing values, and separates features from labels.

        Returns:
            tuple: (X DataFrame, y Series) — features and raw labels.
        """
        if not self.file_path:
            raise ValueError("ERROR: File not found")
        
        self.data = pd.read_csv(self.file_path)

        # Some CSV exports add an unnamed index column — drop it if present
        if self.data.columns[0].startswith('Unnamed'):
            self.data.drop(columns=[self.data.columns[0]], inplace=True)
            
        # Drop any non-feature metadata columns (e.g. IDs, SMILES strings),
        # but always keep the label column even if accidentally listed in ignore_desc
        if self.ignore_desc:
            safe_drop = [c for c in self.ignore_desc if c != self.labels]
            self.data.drop(columns=safe_drop, inplace=True, errors='ignore')
            
        # Print shape and NaN report before any cleaning
        self.diagnostic_analysis()
        # Handle missing values (impute or drop rows depending on severity)
        self._impute_missing_values()
    
        # Separate the label column from the features
        if self.labels in self.data.columns:
            self.lb_data = self.data[self.labels].copy()
            self.data.drop(columns=[self.labels], inplace=True)
        else:
            raise ValueError("ERROR: The label column is missing")
        
        return self.data, self.lb_data

    def diagnostic_analysis(self):
        """
        Prints a summary of the dataset shape and a report of any missing values,
        including whether the NaN rows are the same across all affected columns.
        """
        if self.data is None:
            raise ValueError("ERROR: No data loaded , please load data first")

        M2, N2 = np.shape(self.data)
        print(f'Shape of Data (Before Splits/SMOTE): {M2} rows x {N2} columns')
        print('-------------------------------------------------')

        # Count NaNs per column and filter to only those that have at least one
        nan_count = self.data.isna().sum()
        nan_columns = nan_count[nan_count > 0]

        if not nan_columns.empty:
            print('Nan values were detected.\n')
            # .to_string() prevents pandas from truncating the output with "..."
            print(nan_columns.to_string())
            
            # Check if all NaN values fall on the exact same rows across columns.
            # If total rows-with-any-NaN equals the per-column NaN count (and it's
            # uniform), then it's the same rows — safe to drop once.
            rows_with_any_nans = self.data.isna().any(axis=1).sum()
            print(f'\nTotal rows containing at least one NaN: {rows_with_any_nans}')
            if rows_with_any_nans == nan_columns.max() and len(set(nan_columns)) == 1:
                print('-> Conclusion: The NaN values are in the exact same rows across all affected columns.')
            else:
                print('-> Conclusion: The NaN values are scattered across different rows.')
                
            print('\n-------------------------------------------------')
        else:
            print('No NaN values detected.')
            
    def _impute_missing_values(self):
        """
        Handles NaN values dynamically:
        - If a column's NaNs are >= 10% of total rows, fills with the column median.
        - If a column's NaNs are < 10% of total rows, drops the rows with those NaNs.
        """
        if self.data.isnull().sum().sum() == 0:
            return  # No missing values, skip the process

        total_rows = len(self.data)
        threshold = 0.10 * total_rows  # The 10% cutoff

        # Get counts of NaNs per column
        nan_counts = self.data.isnull().sum()
        cols_with_nans = nan_counts[nan_counts > 0]

        cols_to_impute = []    # Columns where NaNs will be filled with the median
        cols_to_drop_rows = [] # Columns where the affected rows will be dropped entirely

        # Sort columns into the two buckets based on the threshold
        for col, count in cols_with_nans.items():
            if count >= threshold:
                cols_to_impute.append(col)
            else:
                cols_to_drop_rows.append(col)

        # 1. Fill with median for columns >= 10%
        # Imputing preserves all rows when data loss would be too high
        if cols_to_impute:
            print(f"Imputing NaNs with median for columns (>= 10% missing): {cols_to_impute}")
            # Only impute numeric columns to prevent type errors on categorical columns
            numeric_impute_cols = [c for c in cols_to_impute if pd.api.types.is_numeric_dtype(self.data[c])]
            self.data[numeric_impute_cols] = self.data[numeric_impute_cols].fillna(self.data[numeric_impute_cols].median())

        # 2. Drop the rows entirely for columns < 10%
        # When only a small fraction of rows are affected, dropping is safer than imputing
        if cols_to_drop_rows:
            print(f"Dropping rows due to NaNs in columns (< 10% missing): {cols_to_drop_rows}")
            self.data.dropna(subset=cols_to_drop_rows, inplace=True)
            
            # Reset the index cleanly so there are no gaps in the row indices after dropping
            self.data.reset_index(drop=True, inplace=True)

        M3, N3 = np.shape(self.data)
        print(f"Shape of Data (After Imputation): {M3} rows x {N3} columns")
        print('-------------------------------------------------')


    def get_split(self, test_size=0.2, random_state=142):
        """
        Encodes labels and splits data into stratified train/test sets.

        Args:
            test_size (float): Fraction of data to hold out for testing (default 0.2 = 20%).
            random_state (int): Seed for reproducibility.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X = self.data.copy()
        y = self.lb_data.copy()

        # Print unique values first — useful for catching unexpected label names in the CSV
        print(f"Unique label values found in data: {sorted(y.unique())}")

        # Map string labels to integers using the explicit label_map
        y_encoded = y.map(self.label_map)

        # Validate that every label was successfully mapped — catch typos or unknown categories
        unmapped = y[y_encoded.isna()]
        if not unmapped.empty:
            raise ValueError(
                f"ERROR: The following label values could not be mapped: {unmapped.unique().tolist()}\n"
                "Update label_map to include them."
            )

        y_encoded = y_encoded.values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=test_size,random_state=random_state,stratify=y_encoded)
        return X_train, X_test, y_train, y_test


class DataConv:
    """Utility class for data transformation operations (scaling, oversampling)."""

    @staticmethod
    def Scaler(scaler_type='standard'):
        scaler = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }

        if scaler_type not in scaler:
            raise ValueError(f"Scaler type not found. Avaliable scaler: {list(scaler.keys())}")
        
        return scaler[scaler_type]

    @staticmethod
    def get_smote():
        """
        Returns the callable Borderline SMOTE object.
        DO NOT call .fit_resample() here! The ImbPipeline in train.py 
        must do that internally to prevent data leakage.
        """
        return BorderlineSMOTE(random_state=67)

class ModelExporter:
    """Handles the preparation and export of models for production."""

    @staticmethod
    def _slice_scaler(fitted_scaler, support_mask, num_features):
        """Slices a fitted StandardScaler to keep only selected features."""
        sliced_scaler = StandardScaler()
        sliced_scaler.mean_ = fitted_scaler.mean_[support_mask]
        sliced_scaler.scale_ = fitted_scaler.scale_[support_mask]
        sliced_scaler.var_ = fitted_scaler.var_[support_mask]
        sliced_scaler.n_features_in_ = num_features
        
        if hasattr(fitted_scaler, 'feature_names_in_'):
            sliced_scaler.feature_names_in_ = fitted_scaler.feature_names_in_[support_mask]
            
        return sliced_scaler

    @classmethod
    def export_production_pipeline(cls, final_pipeline, support_mask, selected_features, model_name):
        """
        Extracts components from a training pipeline, handles dimension mismatches, 
        and exports a clean inference-only ONNX model and feature manifest.
        """
        import os
        os.makedirs("models", exist_ok=True)
        
        # 1. Save feature manifest for the backend team
        feature_json_path = f"models/Production_Features_{model_name}.json" 
        with open(feature_json_path, "w") as f: 
            json.dump(selected_features, f) 
        print(f"Saved required feature names to: {feature_json_path}")

        # 2. Extract and slice the scaler
        fitted_scaler = final_pipeline.named_steps['scaler'] 
        sliced_scaler = cls._slice_scaler(fitted_scaler, support_mask, len(selected_features))

        # 3. Build a clean inference pipeline
        fitted_classifier = final_pipeline.named_steps['classifier'] 
        inference_pipeline = SklearnPipeline([ 
            ('scaler', sliced_scaler), 
            ('classifier', fitted_classifier) 
        ]) 

        # 4. Convert to ONNX
        # Note: Scikit-learn models like RandomForestClassifier are natively supported by skl2onnx.
        # External models like LightGBM and XGBoost need to be explicitly registered.
        try:
            from skl2onnx import update_registered_converter
            from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
            
            # Register LightGBM
            try:
                import lightgbm
                from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
                update_registered_converter(
                    lightgbm.LGBMClassifier, 'LightGbmLGBMClassifier',
                    calculate_linear_classifier_output_shapes, convert_lightgbm,
                    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
                )
            except (ImportError, ValueError):
                pass
                
            # Register XGBoost
            try:
                import xgboost
                from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
                update_registered_converter(
                    xgboost.XGBClassifier, 'XGBoostXGBClassifier',
                    calculate_linear_classifier_output_shapes, convert_xgboost,
                    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
                )
            except (ImportError, ValueError):
                pass
                
        except Exception:
            pass

        initial_type = [('float_input', FloatTensorType([None, len(selected_features)]))] 
        onnx_model = convert_sklearn(inference_pipeline, initial_types=initial_type, target_opset={'': 12, 'ai.onnx.ml': 3})

        # 5. Save to disk
        onnx_filename = f"models/Production_{model_name}.onnx" 
        with open(onnx_filename, "wb") as f: 
            f.write(onnx_model.SerializeToString()) 
         
        print(f"Architecture complete. Saved ONNX model securely: {onnx_filename}")


if __name__ == "__main__":
    # --- Quick sanity-check run ---
    # Replace CSV_PATH with the actual path to your dataset before running.
    pipeline = DataProcessor(
        file_path=r"CSV_PATH",
        labels="Label",
        ignore_desc=["Molecule ChEMBL ID", "Smiles", "Standard Type", "Standard Relation", "Standard Value", "Standard Units"]
    )
    df = pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.get_split()

    print(f"All done! Shape of X_train:{X_train.shape} , X_test:{X_test.shape} , y_train:{y_train.shape} , y_test:{y_test.shape}")
    print('-------------------------------------------------')
    # Check class balance in both splits — expect a similar Active/Not Active ratio
    print("Original training label distribution:")
    print(pd.Series(y_train).value_counts().to_string())
    print("\nOriginal test label distribution:")
    print(pd.Series(y_test).value_counts().to_string())
