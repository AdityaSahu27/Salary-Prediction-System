from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import warnings

app = Flask(__name__)

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

columns = ['AGE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS', 'PAST EXP', 'TENURE', 'SEX', 'DESIGNATION', 'UNIT']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the features from the request
        data = request.json['features']
        print(f"Received features: {data}")
        
        # Ensure the data is a 2D array
        if not isinstance(data[0], list):
            data = [data]
        
        # Convert the features to a DataFrame
        input_df = pd.DataFrame(data, columns=columns)
        print(f"Input DataFrame:\n{input_df}")
        
        # Suppress the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # Make the prediction
            prediction = model.predict(input_df)
        
        print(f"Prediction: {prediction}")
        
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        # Log any exceptions for debugging
        print("Prediction error:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
