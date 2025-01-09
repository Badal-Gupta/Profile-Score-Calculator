# Profile Score Prediction API

## Overview
This repository contains the code for a Flask-based web API that predicts a user's profile score based on input data. The model used for prediction is a pre-trained linear regression model saved as a pickle file. The API provides endpoints for health checks and predictions.

## Features
- **Prediction**: Provides profile score predictions using a machine learning model.
- **Data Preprocessing**: Includes steps like one-hot encoding for categorical data and TF-IDF vectorization for text data.
- **Modular Design**: Code is modular with clearly defined functions for loading models, preprocessing data, and making predictions.
- **Logging**: Debug-level logging for better traceability.

## Folder Structure
```
root
│
├── inference.py                 # Main Flask application
├── linear-model.pkl       # Pre-trained machine learning model
├── data.csv               # Dataset used for preprocessing
├── README.md              # Documentation (this file)
```

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or later
- pip (Python package installer)

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Badal-Gupta/Profile-Score-Calculator.git
   cd profile-score-api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python inference.py
   ```
   By default, the application runs on `http://0.0.0.0:8090`.

## Endpoints

### 1. Health Check (`/ping`)
**Method**: `GET`

**Description**: Verifies if the model is loaded and the API is functioning.

**Response**:
- `200 OK`: `{ "status": "Healthy" }`
- `404 Not Found`: `{ "status": "Unhealthy" }`

### 2. Prediction (`/invocations`)
**Method**: `POST`

**Description**: Processes input data and returns the predicted profile score.

**Request**:
- **Headers**:
  - `Content-Type: application/json`
- **Body** (example):
  ```json
  {
    "Age": "<35",
    "Accessibility": "No",
    "EdLevel": "Undergraduate",
    "Employment": "1",
    "Gender": "Man",
    "MentalHealth": "No",
    "MainBranch": "Dev",
    "YearsCode": "12",
    "YearsCodePro": "5",
    "Country": "Spain",
    "PreviousSalary": 96482.0,
    "HaveWorkedWith": "Bash/Shell;HTML/CSS;JavaScript",
    "ComputerSkills": "12"
  }
  ```

**Response**:
- Example:
  ```json
  {
    "prediction": 85
  }
  ```

## Key Functions

- **`model_fn(model_dir)`**: Loads the pre-trained model from a specified directory.
- **`input_fn(request_body, request_content_type)`**: Parses and preprocesses the input data.
- **`predict_fn(input_data, model)`**: Makes predictions using the model.
- **`output_fn(prediction, accept)`**: Formats the prediction output.

## File Descriptions

- **`inference.py`**: The main application code containing API endpoints and supporting functions.
- **`linear-model.pkl`**: Pre-trained machine learning model used for predictions.
- **`data.csv`**: Sample dataset used for preprocessing (e.g., fitting encoders and vectorizers).
- **`requirements.txt`**: Lists all required Python libraries.

## Logging
Logging is enabled to trace requests, model loading, and other operations. Logs are printed to the console for debugging purposes.

## Error Handling
The API handles common errors such as:
- Unsupported content type
- Missing model file
- Invalid input data

## Deployment
You can deploy this API using any cloud service (e.g., AWS, Azure, Google Cloud) or containerize it using Docker.

## Future Improvements
- Implement user authentication for secure access.
- Add input validation for stricter error handling.
- Enhance model performance with additional features and training.

## Contributing
Feel free to submit issues or pull requests for improvements. Make sure to follow best practices for Python and Flask development.

## License
This project is licensed under the [MIT License](LICENSE).

