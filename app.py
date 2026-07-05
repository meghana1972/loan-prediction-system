from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('models/best_model.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')

# Feature columns (must match training order)
FEATURE_COLUMNS = ['Gender', 'Married', 'Dependents', 'Education', 
                   'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                   'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

@app.route('/')
def home():
    """Home page - shows the loan prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        gender = request.form.get('gender')
        married = request.form.get('married')
        dependents = request.form.get('dependents')
        education = request.form.get('education')
        self_employed = request.form.get('self_employed')
        applicant_income = float(request.form.get('applicant_income'))
        coapplicant_income = float(request.form.get('coapplicant_income'))
        loan_amount = float(request.form.get('loan_amount'))
        loan_amount_term = float(request.form.get('loan_amount_term'))
        credit_history = float(request.form.get('credit_history'))
        property_area = request.form.get('property_area')
        
        # Create input data
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }])
        
        # Get scaler and label encoders from preprocessor
        scaler = preprocessor['scaler']
        label_encoders = preprocessor['label_encoders']
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        for col in categorical_cols:
            if col in input_data.columns:
                le = label_encoders[col]
                input_data[col] = le.transform(input_data[col].astype(str))
        
        # Handle Dependents
        input_data['Dependents'] = input_data['Dependents'].replace('3+', 3)
        input_data['Dependents'] = input_data['Dependents'].astype(float)
        
        # Scale numerical features
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_data[FEATURE_COLUMNS])
        prediction_proba = model.predict_proba(input_data[FEATURE_COLUMNS])
        
        # Format result
        result = {
            'approved': bool(prediction[0] == 1),
            'probability_approved': float(prediction_proba[0][1]),
            'probability_rejected': float(prediction_proba[0][0])
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')