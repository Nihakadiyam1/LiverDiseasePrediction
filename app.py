import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Liver Disease Prediction App", layout="wide")

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('Liver_disease_data.csv')
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    # One-hot encode the Gender column
    X = pd.get_dummies(X, columns=['Gender'], prefix='Gender', drop_first=True)
    return X, y

X, y = load_data()

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training data
@st.cache_resource
def train_model():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return scaler, model

scaler, model = train_model()

# Evaluate the model on the test data
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.title('Liver Disease Prediction App')
st.write(f"Model Test Accuracy: {accuracy:.2f}")

st.write('Enter the following information to predict liver disease:')


# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0, format="%.1f")
    alcohol_consumption = st.number_input('Alcohol Consumption', min_value=0.0, value=5.0, format="%.1f")
    smoking = st.selectbox('Smoking', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

with col2:
    genetic_risk = st.selectbox('Genetic Risk', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    physical_activity = st.number_input('Physical Activity', min_value=0.0, value=1.0, format="%.1f")
    diabetes = st.selectbox('Diabetes', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    liver_function_test = st.number_input('Liver Function Test', min_value=0.0, value=50.0, format="%.1f")

# Center the predict button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button('Predict', use_container_width=True)

if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'BMI': [bmi],
        'AlcoholConsumption': [alcohol_consumption],
        'Smoking': [smoking],
        'GeneticRisk': [genetic_risk],
        'PhysicalActivity': [physical_activity],
        'Diabetes': [diabetes],
        'Hypertension': [hypertension],
        'LiverFunctionTest': [liver_function_test],
        'Gender': [gender]
    })

    # One-hot encode the Gender column to match the training data
    input_data = pd.get_dummies(input_data, columns=['Gender'], prefix='Gender', drop_first=True)

    # Ensure all columns from training data are present
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[X.columns]

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.warning('The Patient is having liver disease')
    else:
        st.success('The Patient is not having liver disease')
    

st.markdown("---")
