import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle

from imblearn.over_sampling import RandomOverSampler

# Load the synthetic dataset
@st.cache
def load_data():
    df = pd.read_csv("C:/Users/King/Downloads/Projects portfolio/Project 7/heart_disease_app/synthetic_heart_disease_dataset_v4.csv")
    return df

df = load_data()

# Separate the features (X) and target variable (y)
X = df.drop(columns=['HeartDisease'])  # Assuming 'HeartDisease' is the target variable
y = df['HeartDisease']

# Encode categorical variables using one-hot encoding
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
X_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
X = pd.concat([X, X_encoded], axis=1)
X = X.drop(columns=categorical_cols)

# Balance the dataset using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Train the model on the resampled data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('C:/Users/King/Downloads/Projects portfolio/Project 7/heart_disease_app/random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

# Define the feature names used during training (excluding one-hot encoded names)
feature_names = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
                 'ExerciseAngina', 'Oldpeak', 'ChestPainType_ATA', 'ChestPainType_NAP',
                 'ChestPainType_ASY', 'RestingECG_ST', 'RestingECG_LVH', 'ST_Slope_Flat',
                 'ST_Slope_Down']

# Streamlit app title and description
st.title('Heart Disease Prediction App')
st.write('This app predicts the likelihood of heart disease based on user input.')

# Sidebar with user input fields
st.sidebar.header('User Input')

# Define input fields for user data
age = st.sidebar.slider('Age', 30, 80, 50)
sex = st.sidebar.radio('Sex', ['Male', 'Female'])
resting_bp = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
cholesterol = st.sidebar.slider('Cholesterol (mg/dL)', 120, 300, 200)
fasting_bs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dL', ['No', 'Yes'])
max_hr = st.sidebar.slider('Maximum Heart Rate', 60, 202, 150)
exercise_angina = st.sidebar.radio('Exercise-Induced Angina', ['No', 'Yes'])
oldpeak = st.sidebar.slider('Oldpeak ST Depression', 0.0, 6.0, 1.0)

# Add more features
resting_ecg = st.sidebar.radio('Resting ECG', ['Normal', 'ST', 'LVH'])
st_slope = st.sidebar.radio('ST Slope', ['Up', 'Flat', 'Down'])
chest_pain_type = st.sidebar.radio('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])

# Map user inputs to numerical values
sex_mapping = {'Male': 1, 'Female': 0}
fasting_bs_mapping = {'No': 0, 'Yes': 1}
exercise_angina_mapping = {'No': 0, 'Yes': 1}
resting_ecg_mapping = {'Normal': 0, 'ST': 1, 'LVH': 2}
st_slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}
chest_pain_type_mapping = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}

sex = sex_mapping[sex]
fasting_bs = fasting_bs_mapping[fasting_bs]
exercise_angina = exercise_angina_mapping[exercise_angina]
resting_ecg = resting_ecg_mapping[resting_ecg]
st_slope = st_slope_mapping[st_slope]
chest_pain_type = chest_pain_type_mapping[chest_pain_type]

# Create a user data array for prediction with the same features used during training
user_data = np.array([age, sex, resting_bp, cholesterol, fasting_bs, max_hr,
                      exercise_angina, oldpeak, chest_pain_type, resting_bp, cholesterol,
                      fasting_bs, resting_ecg, max_hr, exercise_angina]).reshape(1, -1)

# Predict heart disease probability and display output
if st.button('Predict'):
    if user_data is not None:
        try:
            prediction = model.predict(user_data)
            prediction_proba = model.predict_proba(user_data)

            st.subheader('Prediction Result:')
            st.write(f'Predicted Class: {prediction[0]}')
            st.write(f'Predicted Probability for Class 0: {prediction_proba[0][0]:.4f}')
            st.write(f'Predicted Probability for Class 1: {prediction_proba[0][1]:.4f}')
        except Exception as e:
            st.warning('An error occurred during prediction. Please check your input values.')

# Display the entire DataFrame for reference
st.subheader('Synthetic Heart Disease Dataset')
st.write(df)

# Context and Attribute Information
st.sidebar.header('Context')
st.sidebar.write("Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide.")
st.sidebar.write("Four out of 5 CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age.")
st.sidebar.write("Heart failure is a common event caused by CVDs, and this dataset contains 11 features that can be used to predict possible heart disease.")
st.sidebar.write("People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia, or already established disease) need early detection and management, where a machine learning model can be of great help.")

# Attribute Information
st.sidebar.header('Attribute Information')
st.sidebar.write("Age: age of the patient [years]")
st.sidebar.write("Sex: sex of the patient [M: Male, F: Female]")
st.sidebar.write("ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]")
st.sidebar.write("RestingBP: resting blood pressure [mm Hg]")
st.sidebar.write("Cholesterol: serum cholesterol [mm/dl]")
st.sidebar.write("FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]")
st.sidebar.write("RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]")
st.sidebar.write("MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]")
st.sidebar.write("ExerciseAngina: exercise-induced angina [Y: Yes, N: No]")
st.sidebar.write("Oldpeak: oldpeak = ST [Numeric value measured in depression]")
st.sidebar.write("ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]")
st.sidebar.write("HeartDisease: output class [1: heart disease, 0: Normal]")
