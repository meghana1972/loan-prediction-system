import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load the dataset"""
        df = pd.read_csv(filepath)
        # Convert empty strings to NaN
        df = df.replace('', np.nan)
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        
        # Fill categorical missing values with mode
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 
                           'Credit_History', 'Loan_Amount_Term']
        
        for col in categorical_cols:
            if col in df_clean.columns:
                mode_value = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_value)
        
        # Fill numerical missing values with median
        numerical_cols = ['LoanAmount']
        for col in numerical_cols:
            if col in df_clean.columns:
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)
        
        # Handle any remaining NaN values in all columns
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                if not df_clean[col].mode().empty:
                    mode_val = df_clean[col].mode()[0]
                else:
                    mode_val = 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
            else:
                df_clean[col] = df_clean[col].fillna(0)
        
        return df_clean
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Handle Dependents specifically
        if 'Dependents' in df_encoded.columns:
            df_encoded['Dependents'] = df_encoded['Dependents'].replace('3+', 3)
            df_encoded['Dependents'] = df_encoded['Dependents'].astype(float)
        
        # Encode other categorical columns
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 
                           'Property_Area']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        
        # Ensure all numerical columns exist
        for col in numerical_cols:
            if col not in df.columns:
                df[col] = 0
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def prepare_data(self, df, target_col='Loan_Status', test_size=0.2):
        """Complete data preparation pipeline"""
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical
        df = self.encode_categorical(df)
        
        # Scale features
        df = self.scale_features(df)
        
        # Separate features and target
        X = df.drop(columns=[target_col, 'Loan_ID'])
        y = df[target_col].map({'Y': 1, 'N': 0})
        
        # FINAL SAFETY CHECK: Replace any remaining NaN with 0
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, df
    
    def save_preprocessor(self, filepath='models/preprocessor.joblib'):
        """Save the preprocessor objects"""
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, filepath)
    
    def load_preprocessor(self, filepath='models/preprocessor.joblib'):
        """Load the preprocessor objects"""
        preprocessor = joblib.load(filepath)
        self.scaler = preprocessor['scaler']
        self.label_encoders = preprocessor['label_encoders']