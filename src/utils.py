# utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    """
    # Replace missing values with the median for numerical features
    data.fillna(data.median(), inplace=True)
    return data

def encode_categorical_variables(data):
    """
    Encode categorical variables into numerical values.
    """
    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data, label_encoders

def scale_numerical_features(data):
    """
    Scale numerical features to have zero mean and unit variance.
    """
    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['int', 'float']).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data, scaler
