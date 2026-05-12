import os
import json
import argparse
import numpy as np
import pandas as pd
import warnings
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import onnxruntime as ort

warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')

class LocalModelTester:
    @staticmethod
    def get_smile_columns(df):
        columns = df.columns.to_list()

        # Tier 1: Lexical Heuristics
        common_names = {'smiles', 'canonical_smiles', 'structure', 'zincsmiles', 'compound_smiles'}

        for col in columns:
            column_filt = str(col).strip().lower()
            if column_filt in common_names:
                return col

        # Tier 2: Empirical test
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        
        if not text_columns.empty:
            sample_df = df.head(10)
            smile_col = None
            highest_num = 0
            
            for col in text_columns:
                sample = sample_df[col].dropna()
                valid_str = 0
                actual_num_row = len(sample)
                if actual_num_row == 0:
                    continue

                for items in sample:
                    if Chem.MolFromSmiles(str(items)) is not None:
                        valid_str += 1
                        
                # Require at least 50% to be valid chemistry
                required_matches = max(1, int(actual_num_row * 0.5))        
                
                if valid_str > highest_num and valid_str >= required_matches:
                    highest_num = valid_str
                    smile_col = col
        
            if smile_col:
                return smile_col
                
        raise ValueError("Could not automatically detect a valid SMILES column. Please name your structure column 'SMILES'.")
    
    @staticmethod
    def get_id(df, smile_col):
        columns = df.columns.to_list()

        # Common variations of Mol_id or typical ID columns
        id_patterns = {
            'mol_id', 'molid', 'id', 'zinc_id', 'zincid', 'zinc_ids', 
            'compound_id', 'compoundid', 'compound_name', 'name', 
            'title', 'structure_id'
        }

        # Priority 1: Exact matches (after normalization)
        for col in columns:
            column_filt = str(col).strip().lower()
            if column_filt in id_patterns and col != smile_col:
                return col
            
        return None
    
    @classmethod
    def run_prediction(cls, csv_file, model_path, descriptors_path, output_file):
        print(f"[*] Loading descriptors from: {descriptors_path}")
        with open(descriptors_path, 'r') as f:
            descriptors = json.load(f)
            
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
        
        print(f"[*] Loading ONNX model from: {model_path}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        print(f"[*] Reading data from: {csv_file}")
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except Exception as e:
            raise ValueError(f"Failed to read the CSV file. ERROR: {str(e)}")

        if df.empty:
            raise ValueError("The CSV file is empty.")
        
        smile_column = cls.get_smile_columns(df)
        id_column = cls.get_id(df, smile_column)
        
        print(f"[*] Detected SMILES column: '{smile_column}'")
        if id_column:
            print(f"[*] Detected ID column: '{id_column}'")

        valid_rows = []
        valid_mol = []

        print("[*] Parsing chemistry...")
        for idx, rows in df.iterrows():
            mol = Chem.MolFromSmiles(str(rows[smile_column]))
            if mol:
                valid_mol.append(Chem.AddHs(mol))
                valid_rows.append(rows)

        if not valid_mol:
            raise ValueError("NO valid SMILE string found in the file.")        
          
        valid_smiles_df = pd.DataFrame(valid_rows)
        
        print(f"[*] Calculating {len(descriptors)} RDKit descriptors for {len(valid_mol)} molecules...")
        calculate_descriptors = [calc.CalcDescriptors(mol) for mol in valid_mol]
        descriptors_df = pd.DataFrame(
            calculate_descriptors,
            columns=descriptors,
            index=valid_smiles_df.index,
        )

        numeric_cols = descriptors_df.select_dtypes(include=['number']).columns
        # Fill missing with median
        descriptors_df[numeric_cols] = descriptors_df[numeric_cols].fillna(descriptors_df[numeric_cols].median())

        print("[*] Running ONNX inference...")
        X = descriptors_df[descriptors].values.astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X})
        
        # Extract probabilities
        prob = outputs[1]
        confidence_scores = np.array([p.get(1, 0.0) for p in prob])
    
        # Formatting Output
        rename_map = {smile_column: 'SMILES'}
        if id_column: 
            rename_map[id_column] = 'ID'

        cols_keep = [smile_column]
        if id_column:
            cols_keep.append(id_column) 

        results = valid_smiles_df[cols_keep].copy()
        results = results.rename(columns=rename_map)

        results['Predicted_Activity'] = np.where(confidence_scores >= 0.6, 'Active', 'Not Active')
        results['Confidence'] = confidence_scores

        print(f"[*] Saving results to: {output_file}")
        results.to_csv(output_file, index=False)
        print("[*] Done!")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locally test an ONNX molecular prediction pipeline.")
    
    parser.add_argument('--csv', required=True, help="Path to the input CSV file containing SMILES.")
    parser.add_argument('--model', required=True, help="Path to the trained .onnx model file.")
    parser.add_argument('--desc', required=True, help="Path to the JSON file containing descriptor names.")
    parser.add_argument('--out', default="predictions_output.csv", help="Path to save the output CSV. (Default: predictions_output.csv)")

    args = parser.parse_args()
    
    LocalModelTester.run_prediction(
        csv_file=args.csv,
        model_path=args.model,
        descriptors_path=args.desc,
        output_file=args.out
    )