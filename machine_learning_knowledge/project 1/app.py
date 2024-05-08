from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained random forest regressor model
random_forest_regressor = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    input_data = request.json.get('input_data')

    # Convert input data to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = random_forest_regressor.predict(input_data_reshaped)

    # Return prediction as JSON response
    return jsonify({'prediction': prediction[0]}), 200

if __name__ == '__main__':
    app.run(debug=True)
