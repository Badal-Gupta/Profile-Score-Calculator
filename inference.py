import os
import pickle
import json
import numpy as np
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from flask import Flask, request, jsonify

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

app = Flask(__name__)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def model_fn(model_dir):
    """
    Load the model from the specified directory.
    """
    model_path = os.path.join(model_dir, 'linear-model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file does not exist: {model_path}')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    logger.info("Model loaded successfully")
    return model

def input_fn(request_body, request_content_type='application/json'):
    """
    Process the input data from the request body.
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        df = pd.DataFrame([input_data])
        DF = pd.read_csv('data.csv')
        DF.dropna(inplace=True)
        X = DF.drop(['Employed', 'Unnamed: 0'], axis=1)
        # Define categorical columns
        categorical_cols = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Country']

        # Preprocessing pipelines
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

        # Fit vectorizer and encoder separately for feature name extraction
        vectorizer.fit(X['HaveWorkedWith'])
        one_hot_encoder.fit(X[categorical_cols])

        # Apply transformations using ColumnTransformer
        transformer = ColumnTransformer([
            ('vectorizer', vectorizer, 'HaveWorkedWith'),
            ('encoder', one_hot_encoder, categorical_cols)
        ])

        X = transformer.fit_transform(X)
        df = transformer.transform(df)

        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make a prediction using the provided model and input data.
    """
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, accept='application/json'):
    """
    Format the prediction output as specified.
    """
    response = {'prediction': int(sigmoid(prediction[0])*100)}
    if accept == 'application/json':
        return json.dumps(response), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

# Load the model
model_dir = ''
model = model_fn(model_dir)

@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint to verify if the model is loaded.
    """
    health = model is not None
    status = 200 if health else 404
    return jsonify({'status': 'Healthy' if health else 'Unhealthy'}), status

@app.route('/invocations', methods=['POST'])
def invoke():
    """
    Endpoint to process incoming requests and return predictions.
    """
    data = request.data.decode('utf-8')
    content_type = request.content_type
    
    # Process input data
    input_data = input_fn(data, content_type)

    # Make a prediction
    prediction = predict_fn(input_data, model)

    # Format the output
    response, content_type = output_fn(prediction, content_type)
    
    return response, 200, {'Content-Type': content_type}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)