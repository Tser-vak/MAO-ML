import pandas as pd
import numpy as np 
import imblearn.over_sampling as BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder


class DataProcessor:
    def __init__(self,file_path , ignore_desc=None , labels=None):
        self.file_path = file_path
        self.ignore_desc = ignore_desc if ignore_desc is not None else []

        if labels is not None:
            self.labels = labels
        else:
            raise ValueError("WARNING: The label column is missing")
        
        self.lb_data = None
        self.data = None
        self.label_map = {'Not Active': 0, 'Active': 1}

    def load_data(self):

        if not self.file_path:
            raise ValueError("ERROR: File not found")
        
        self.data = pd.read_csv(self.file_path)

        if self.data.columns[0].startswith('Unnamed'):
            self.data.drop(columns=[self.data.columns[0]], inplace=True)
            
        if self.ignore_desc:
            safe_drop = [c for c in self.ignore_desc if c != self.labels]
            self.data.drop(columns=safe_drop, inplace=True, errors='ignore')
            
        self.diagnostic_analysis()
        self._impute_missing_values()
        
    
        if self.labels in self.data.columns:
            self.lb_data = self.data[self.labels].copy()
            self.data.drop(columns=[self.labels], inplace=True)
        else:
            raise ValueError("ERROR: The label column is missing")
        
        return self.data , self.lb_data

    def diagnostic_analysis(self):

        if self.data is None:
            raise ValueError("ERROR: No data loaded , please load data first")

        M2,N2 = np.shape(self.data)
        print(f'Shape of Data (Before Splits/SMOTE): {M2} rows x {N2} columns')
        print('-------------------------------------------------')

        nan_count = self.data.isna().sum()
        nan_columns = nan_count[nan_count > 0]
        if not nan_columns.empty:
            print('Nan values were detected.\n')
            print(nan_columns.to_string())
            
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
        threshold = 0.10 * total_rows  # Calculate the 10% mark

        # Get counts of NaNs per column
        nan_counts = self.data.isnull().sum()
        cols_with_nans = nan_counts[nan_counts > 0]

        cols_to_impute = []
        cols_to_drop_rows = []

        # Sort columns into the two buckets based on the threshold
        for col, count in cols_with_nans.items():
            if count >= threshold:
                cols_to_impute.append(col)
            else:
                cols_to_drop_rows.append(col)

        # 1. Fill with median for columns >= 10%
        if cols_to_impute:
            print(f"Imputing NaNs with median for columns (>= 10% missing): {cols_to_impute}")
            # Ensure we only impute numeric columns to prevent errors
            numeric_impute_cols = [c for c in cols_to_impute if pd.api.types.is_numeric_dtype(self.data[c])]
            self.data[numeric_impute_cols] = self.data[numeric_impute_cols].fillna(self.data[numeric_impute_cols].median())

        # 2. Drop the rows entirely for columns < 10%
        if cols_to_drop_rows:
            print(f"Dropping rows due to NaNs in columns (< 10% missing): {cols_to_drop_rows}")
            self.data.dropna(subset=cols_to_drop_rows, inplace=True)
            
            # Reset the index cleanly so there are no skipped numbers in the row indices
            self.data.reset_index(drop=True, inplace=True)
        M3,N3 = np.shape(self.data)
        print(f"Shape of Data (After Imputation): {M3} rows x {N3} columns")
        print('-------------------------------------------------')
        

    def get_split(self,test_size=0.2,random_state=142):

        X = self.data.copy()
        y = self.lb_data.copy()

        print(f"Unique label values found in data: {sorted(y.unique())}")
        y_encoded = y.map(self.label_map)

        unmapped = y[y_encoded.isna()]
        if not unmapped.empty:
            raise ValueError(f"ERROR: The following label values could not be mapped: {unmapped.unique().tolist()}\nUpdate label_map to include them.")

        y_encoded = y_encoded.values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=test_size,random_state=random_state,stratify=y_encoded)
        return X_train, X_test, y_train, y_test

class DataConv:

    @staticmethod
    def Scaler(scaler_type='standard'):
        scaler = {
            'standard':StandardScaler(),
            'minmax':MinMaxScaler()
        }

        if scaler_type not in scaler:
            raise ValueError(f"Scaler type not found. Avaliable scaler: {list(scaler.keys())}")
        
        return scaler[scaler_type]
    @staticmethod
    def apply_borderline_smote(X,y,sampling_method='borderline_smote',random_state=142):
        """Applies BorderlineSMOTE strictly to the training data."""
        print("Applying BorderlineSMOTE to training data...")
        smote = BorderlineSMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"New training feature shape after SMOTE: {X_resampled.shape}")
        return X_resampled, y_resampled

if __name__ == "__main__":
    pipeline = DataProcessor(file_path=r"CSV_PATH", labels="Label", ignore_desc=["Molecule ChEMBL ID","Smiles","Standard Type","Standard Relation","Standard Value","Standard Units"])
    df=pipeline.load_data()
    X_train, X_test, y_train, y_test = pipeline.get_split()

    print(f"All done! Shape of X_train:{X_train.shape} , X_test:{X_test.shape} , y_train:{y_train.shape} , y_test:{y_test.shape}")
    print('-------------------------------------------------')
    print("Original training label distribution:")
    print(pd.Series(y_train).value_counts().to_string())
    print("\nOriginal test label distribution:")
    print(pd.Series(y_test).value_counts().to_string())
