from src.data_preprocessing import DataPreprocessor
from src.model_utils import ModelTrainer
import os

def main():
    print("=" * 50)
    print("🏦 LOAN PREDICTION MODEL TRAINING")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    print("\n📊 Loading and preparing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/loan_prediction.csv')
    print(f"✅ Loaded {len(df)} records")
    
    X_train, X_test, y_train, y_test, _ = preprocessor.prepare_data(df)
    print(f"✅ Training set: {len(X_train)} records")
    print(f"✅ Test set: {len(X_test)} records")
    
    # Step 2: Save preprocessor
    print("\n💾 Saving preprocessor...")
    preprocessor.save_preprocessor()
    print("✅ Preprocessor saved!")
    
    # Step 3: Train models
    print("\n🤖 Training models...")
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train)
    
    # Step 4: Evaluate models
    print("\n📈 Evaluating models...")
    trainer.evaluate_models(X_test, y_test)
    
    # Step 5: Find and save best model
    print("\n🏆 Finding best model...")
    best_model, best_name = trainer.find_best_model()
    trainer.save_model(best_model, 'models/best_model.joblib')
    
    print("\n" + "=" * 50)
    print(f"✅ TRAINING COMPLETE! Best model: {best_name}")
    print("=" * 50)

if __name__ == "__main__":
    main()