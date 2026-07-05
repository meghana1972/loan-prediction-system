import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_models(self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        
        return self.trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"\n{name} Accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return self.results
    
    def find_best_model(self):
        """Find the best performing model"""
        best_accuracy = 0
        for name, results in self.results.items():
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                self.best_model = self.trained_models[name]
                self.best_model_name = name
        
        print(f"\n🏆 Best Model: {self.best_model_name} with accuracy: {best_accuracy:.4f}")
        return self.best_model, self.best_model_name
    
    def save_model(self, model, filename='models/best_model.joblib'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, filename)
        print(f"✅ Model saved as {filename}")
    
    def load_model(self, filename='models/best_model.joblib'):
        """Load a saved model"""
        return joblib.load(filename)
    
    def plot_confusion_matrix(self, model_name, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        os.makedirs('static', exist_ok=True)
        plt.savefig(f'static/confusion_matrix_{model_name}.png')
        plt.close()