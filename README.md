Cancer Diagnosis Prediction System

This project aims to predict whether a cancer diagnosis is Malignant or Benign using machine learning. The user inputs the values for various features (such as tumor size, texture, and others), and the system provides a prediction based on a pre-trained model. The system uses Logistic Regression (or other classification algorithms) for classification, with Grid Search for hyperparameter tuning.

Table of Contents

Overview
Features
Getting Started
Prerequisites
Installation
Dataset
Model
Usage
How It Works
Evaluation
Contributing
License
Acknowledgements
Overview

This project provides a system for predicting whether a cancer diagnosis is Malignant or Benign based on input data. The user is prompted to input values for various features related to the cancerous cells, and the system outputs whether the cancer is malignant (harmful) or benign (non-threatening). The underlying model uses machine learning algorithms like Logistic Regression, Grid Search, and Label Encoding for accurate predictions.

Features

User Input: Allows users to enter feature values for cancer diagnosis prediction.
Model Prediction: Uses a pre-trained model to predict the diagnosis.
Easy to Use: Intuitive user interface for input and prediction display.
Preprocessing: Standardizes the input data using a pre-defined scaler.
Accuracy: Provides a clear message based on the prediction (Malignant or Benign).
Getting Started

Prerequisites
Make sure you have the following Python packages installed:

pandas
numpy
matplotlib
seaborn
joblib
scikit-learn
You can install these dependencies using pip:

pip install pandas numpy matplotlib seaborn joblib scikit-learn
Installation
Clone the repository to your local machine:

git clone https://github.com/your-username/cancer-diagnosis-prediction.git
cd cancer-diagnosis-prediction
Dataset

This project assumes the presence of a dataset containing features related to cancer, such as tumor size, texture, and other related factors. For simplicity, we are assuming the dataset is preprocessed and stored in a variable Featured_df. The dataset should contain columns like:

radius_mean
texture_mean
smoothness_mean
compactness_mean
diagnosis (the target column with labels: 'M' for Malignant and 'B' for Benign)
The diagnosis column is excluded from the input feature set, and the dataset is used for training the model.

Model

This project uses a pre-trained classification model that has been tuned using Grid Search for better performance. The model is saved using joblib and loaded for prediction.

Key Model Components:
Label Encoder (LB): Encodes the categorical target labels ('M' and 'B') into numerical values.
Scaler (sc): Standardizes the input data to improve model performance.
Model (grid_search_model): The trained classification model that predicts the diagnosis (Malignant or Benign).
Usage

Input Features: The user will be prompted to enter values for various features of the cancer dataset. For example:
ENTER THE VALUES FOR radius_mean: 14.2
ENTER THE VALUES FOR texture_mean: 19.5
ENTER THE VALUES FOR smoothness_mean: 0.093
...
Prediction Output: Based on the entered values, the model will provide a prediction:
If the model predicts Malignant (M), the output will be:
THE DIAGNOSIS CANCER IS MALIGNANT CONSULT A DOCTOR IMMEDIATELY
If the model predicts Benign (B), the output will be:
THE DIAGNOSIS CANCER IS BENIGN NO NEED TO PANIC
How It Works

User Input: The user is asked to enter the values for various cancer-related features (except for the diagnosis).
Preprocessing: The entered data is preprocessed using a Scaler (standardization) to match the format used in the model training phase.
Prediction: The preprocessed data is passed into the trained model (grid_search_model), which outputs a prediction (Malignant or Benign).
Output Message: Depending on the prediction, an appropriate message is displayed to the user.
df1 = pd.DataFrame([INPUT__DATA], columns=[i for i in Featured_df.columns if i != 'diagnosis'])
scalar_data_input = sc.transform(df1)
prediction_output = grid_serach_mode.predict(scalar_data_input)
original_output = LB.inverse_transform(prediction_output)

if original_output == 'M':
    print(f'THE DIAGNOSIS CANCER IS MALIGNANT CONSULT A DOCTOR IMMEDIATELY')
elif original_output == 'B':
    print(f'THE DIAGNOSIS CANCER IS BENIGN NO NEED TO PANIC')
Evaluation

The model’s performance was evaluated based on standard metrics such as accuracy and precision during the training phase. The model was optimized using Grid Search to find the best hyperparameters, ensuring high accuracy for cancer classification.

Contributing

We welcome contributions to improve this project. To contribute:

Fork the repository.
Create a new branch.
Make your changes.
Submit a pull request.
Please make sure to write tests for any new functionality and ensure that the code is well-documented.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

## Live Demo
Check out the website here: [Predictive Care](https://predictivecare-advanced-feature-3gca.onrender.com/#c7f2025c)
