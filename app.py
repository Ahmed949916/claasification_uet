import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Housing.csv')
    return df

df = load_data()

# Prepare data
X = df.drop("price", axis=1)
y = df["price"]

# Convert categorical variables to numeric
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit UI
st.title("House Price Prediction")

# User inputs for prediction
st.sidebar.header("Enter House Features")

def user_input_features():
    feature_inputs = {}
    for col in X.columns:
        if X[col].dtype == 'float64' or X[col].dtype == 'int64':
            feature_inputs[col] = st.sidebar.number_input(f"Enter {col}", value=float(X[col].mean()))
        else:
            feature_inputs[col] = st.sidebar.radio(f"Select {col}", ["No", "Yes"])
            feature_inputs[col] = 1 if feature_inputs[col] == "Yes" else 0

    return pd.DataFrame([feature_inputs])

input_df = user_input_features()

# Predict button
if st.sidebar.button("Predict Price"):
    prediction_linear = linear_model.predict(input_df)
    prediction_rf = rf_model.predict(input_df)

    st.subheader("Predicted Prices:")
    st.write(f"Linear Regression Prediction: ${prediction_linear[0]:,.2f}")
    st.write(f"Random Forest Prediction: ${prediction_rf[0]:,.2f}")

st.write("Adjust the features on the left panel to predict the house price.")

