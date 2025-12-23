import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="KNN Prediction App", layout="centered")

st.title("🔍 KNN Model Prediction App")

# Load model
@st.cache_resource
def load_model():
    with open("Knn1 (1).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("✅ KNN model loaded successfully")

st.write("### Enter input values")

# 👉 CHANGE feature count based on your model
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(input_data)

    st.subheader("📊 Prediction Result")
    st.write(prediction[0])
