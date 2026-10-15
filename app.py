import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Global variables for model, scaler, and training stats
model = None
scaler = None
metadata = None

def load_assets():
    global model, scaler, metadata
    try:
        if os.path.exists('models/best_model.pkl') and os.path.exists('models/scaler.pkl'):
            model = joblib.load('models/best_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            print("Model and scaler successfully loaded.")
        else:
            print("Warning: Model or scaler files not found. Run train_model.py first.")
            
        if os.path.exists('models/comparison_metrics.json'):
            with open('models/comparison_metrics.json', 'r') as f:
                metadata = json.load(f)
            print("Training metadata loaded successfully.")
        else:
            print("Warning: comparison_metrics.json not found. Run train_model.py first.")
    except Exception as e:
        print(f"Error loading assets: {str(e)}")

# Initial load of model objects
load_assets()

@app.route('/')
def index():
    # If model is not loaded yet, try to load again
    if model is None:
        load_assets()
        
    if metadata is None:
        return "Model and metrics are not initialized. Please run 'python train_model.py' first."
        
    return render_template('index.html', 
                           best_model_name=metadata['best_model_name'],
                           metrics=metadata['metrics'],
                           dataset_summary=metadata['dataset_summary'])

@app.route('/comparison')
def comparison():
    if model is None:
        load_assets()
        
    if metadata is None:
        return "Model and metrics are not initialized. Please run 'python train_model.py' first."
        
    return render_template('comparison.html',
                           best_model_name=metadata['best_model_name'],
                           metrics=metadata['metrics'])

@app.route('/analytics')
def analytics():
    if metadata is None:
        load_assets()
    if metadata is None:
        return "Analytics metrics are not initialized. Please run 'python train_model.py' first."
        
    return render_template('analytics.html')

@app.route('/predict')
def predict():
    if model is None:
        load_assets()
        
    if metadata is None:
        return "Model is not initialized. Please run 'python train_model.py' first."
        
    return render_template('predict.html', 
                           best_model_name=metadata['best_model_name'])

@app.route('/result', methods=['POST'])
def result():
    if model is None or scaler is None or metadata is None:
        load_assets()
        if model is None or scaler is None or metadata is None:
            return "Application assets not loaded. Please train the model first."

    try:
        # 1. Collect form inputs
        pregnancies = float(request.form.get('pregnancies', 0))
        glucose = float(request.form.get('glucose', 0))
        bloodpressure = float(request.form.get('bloodpressure', 0))
        skinthickness = float(request.form.get('skinthickness', 0))
        insulin = float(request.form.get('insulin', 0))
        bmi = float(request.form.get('bmi', 0))
        dpf = float(request.form.get('dpf', 0))
        age = float(request.form.get('age', 0))
        
        # Save raw input dictionary to show to the user
        input_vitals = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bloodpressure,
            'SkinThickness': skinthickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        # 2. Impute invalid 0 values using training medians
        # Fields where 0 is invalid
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        impute_values = metadata['imputation_medians']
        
        processed_vitals = input_vitals.copy()
        was_imputed = {col: False for col in zero_cols}
        
        for col in zero_cols:
            if processed_vitals[col] == 0:
                processed_vitals[col] = impute_values[col]
                was_imputed[col] = True

        # 3. Create model input vector and apply scaling
        input_vector = np.array([[
            processed_vitals['Pregnancies'],
            processed_vitals['Glucose'],
            processed_vitals['BloodPressure'],
            processed_vitals['SkinThickness'],
            processed_vitals['Insulin'],
            processed_vitals['BMI'],
            processed_vitals['DiabetesPedigreeFunction'],
            processed_vitals['Age']
        ]])
        
        scaled_input = scaler.transform(input_vector)
        
        # 4. Predict
        prediction = int(model.predict(scaled_input)[0])
        
        # 5. Calculate probability and confidence score
        # Using predict_proba
        try:
            probabilities = model.predict_proba(scaled_input)[0]
            prob_diabetic = float(probabilities[1]) * 100
            confidence = float(probabilities[prediction]) * 100
        except AttributeError:
            # Fallback if a model does not support predict_proba
            # Decision Tree and KNN do support it, so it should be fine
            prob_diabetic = 100.0 if prediction == 1 else 0.0
            confidence = 100.0
            
        # 6. Risk Classification
        # 0–40% -> Low Risk
        # 41–70% -> Medium Risk
        # 71–100% -> High Risk
        if prob_diabetic <= 40.0:
            risk_class = 'Low Risk'
        elif prob_diabetic <= 70.0:
            risk_class = 'Medium Risk'
        else:
            risk_class = 'High Risk'
            
        return render_template('result.html',
                               prediction=prediction,
                               confidence=confidence,
                               prob_diabetic=prob_diabetic,
                               risk_class=risk_class,
                               input_vitals=input_vitals,
                               impute_values=impute_values,
                               was_imputed=was_imputed,
                               best_model_name=metadata['best_model_name'])
                               
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
