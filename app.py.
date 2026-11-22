import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Breast Cancer Prediction")
st.write("Predict whether a tumour is Benign or Malignant using Machine Learning.")

# Load dataset
@st.cache_resource
def train_model():

    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, data.feature_names, data.target_names


model, accuracy, features, target_names = train_model()

# Display model accuracy
st.subheader("Model Performance")
st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

# User input
st.subheader("Enter Tumour Measurements")

input_values = []

for feature in features[:10]:

    value = st.number_input(
        feature.replace(" ", " ").title(),
        min_value=0.0,
        value=0.0,
        format="%.4f"
    )

    input_values.append(value)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame(
        [input_values],
        columns=features[:10]
    )

    # Fill remaining features with their dataset means
    full_input = pd.DataFrame(
        [model.feature_names_in_],
        columns=features
    )

    full_input.loc[0] = 0

    full_input.loc[0, features[:10]] = input_values

    prediction = model.predict(full_input)
    probabilities = model.predict_proba(full_input)

    predicted_class = target_names[prediction[0]]

    st.subheader("Prediction Result")

    if predicted_class == "benign":
        st.success("The predicted tumour is BENIGN.")
    else:
        st.error("The predicted tumour is MALIGNANT.")

    st.subheader("Prediction Confidence")

    for name, probability in zip(model.classes_, probabilities[0]):

        class_name = target_names[name]

        st.write(
            f"{class_name.title()}: {probability * 100:.2f}%"
        )

        st.progress(float(probability))
