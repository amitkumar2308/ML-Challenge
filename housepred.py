import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Define relative path to dataset
file_path = os.path.join('dataset','Housing.csv')

# Load the dataset
df = pd.read_csv(file_path)

# Define features and target
X = df.drop(columns=['price'])
y = df['price']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea',
                    'furnishingstatus']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Streamlit app
st.title('Housing Price Prediction')

# Sidebar for user input
st.sidebar.header('User Input')


def user_input_features():
    area = st.sidebar.number_input('Area', min_value=0)
    bedrooms = st.sidebar.number_input('Bedrooms', min_value=0)
    bathrooms = st.sidebar.number_input('Bathrooms', min_value=0)
    stories = st.sidebar.number_input('Stories', min_value=0)
    parking = st.sidebar.number_input('Parking', min_value=0)

    mainroad = st.sidebar.selectbox('Mainroad', ['yes', 'no'])
    guestroom = st.sidebar.selectbox('Guestroom', ['yes', 'no'])
    basement = st.sidebar.selectbox('Basement', ['yes', 'no'])
    hotwaterheating = st.sidebar.selectbox('Hotwaterheating', ['yes', 'no'])
    airconditioning = st.sidebar.selectbox('Airconditioning', ['yes', 'no'])
    prefarea = st.sidebar.selectbox('Prefarea', ['yes', 'no'])
    furnishingstatus = st.sidebar.selectbox('Furnishingstatus', ['furnished', 'semi-furnished', 'unfurnished'])

    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict and calculate metrics
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display results
st.write(f"Mean Squared Error: {mse}")

# Prediction on user input
user_pred = pipeline.predict(input_df)
st.write(f"Predicted Price: {user_pred[0]:,.2f}")

# Scatter plot
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Actual vs Predicted Values')

# Add a reference line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

st.pyplot(fig)
