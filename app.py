import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import io

# Load model once
model = load_model("dog_cat_classification.h5")

st.title("🐾 Dog vs Cat Classifier")

# Streamlit Form
with st.form("image_form"):
    st.markdown("**Choose an image by:**")
    uploaded_file = st.file_uploader("📁 Upload from gallery", type=["jpg", "jpeg", "png", "webp"])
    captured_file = st.camera_input("📷 Or click a photo")

    submit_button = st.form_submit_button(label="Predict")

# Choose between upload or capture
image_data = None
if submit_button:
    if uploaded_file is not None:
        image_data = uploaded_file
    elif captured_file is not None:
        image_data = captured_file

    if image_data is not None:
        # Load image
        image = Image.open(image_data)
        st.image(image, caption="Selected Image", use_column_width=True)

        # Convert to array & preprocess
        img = np.array(image)
        img = cv2.resize(img, (256, 256))
        img = img.reshape((1, 256, 256, 3))
        # Normalize if needed
        # img = img / 255.0

        # Predict
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)

        # Show result
        if int(prediction[0][0]) == 0:
            st.success("🧠 It's a **Cat**!")
        else:
            st.success("🧠 It's a **Dog**!")
    else:
        st.warning("⚠️ Please upload or capture an image before submitting.")
