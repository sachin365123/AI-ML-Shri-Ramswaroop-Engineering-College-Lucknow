import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("cat_dog_model.pkl")

IMG_SIZE = 64

st.title("🐱 Cat vs Dog Image Classifier")
st.write("Upload an image to predict whether it is a Cat or Dog.")

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image using Pillow
    image = Image.open(uploaded_file)

    # Convert to RGB (important if image is grayscale/RGBA)
    image = image.convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", width=300)

    # Resize image
    resized = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to NumPy array
    resized = np.array(resized)

    # Flatten image for Logistic Regression
    resized = resized.flatten()

    # Prediction
    prediction = model.predict([resized])[0]

    probability = model.predict_proba([resized])[0]

    # Display prediction
    if prediction == 0:
        st.success("🐱 Prediction: CAT")
    else:
        st.success("🐶 Prediction: DOG")

    # Display probabilities
    st.subheader("Prediction Confidence")

    st.write(f"🐱 Cat Probability: **{probability[0] * 100:.2f}%**")
    st.write(f"🐶 Dog Probability: **{probability[1] * 100:.2f}%**")