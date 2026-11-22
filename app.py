import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data[:, :10], columns=data.feature_names[:10])
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit interface
st.title("Breast Cancer Prediction")
st.write("Random Forest Classification")

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# User inputs
values = []

for feature in X.columns:
    values.append(st.number_input(feature, value=float(X[feature].median())))

# Prediction
if st.button("Predict"):

    result = model.predict(pd.DataFrame([values], columns=X.columns))[0]
    probability = model.predict_proba(pd.DataFrame([values], columns=X.columns))[0]

    if result == 1:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")

    st.write("Benign Confidence:", round(probability[1] * 100, 2), "%")
    st.write("Malignant Confidence:", round(probability[0] * 100, 2), "%")
