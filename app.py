import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Clothing Classifier",
    page_icon="👕",
    layout="centered"
)

st.title("Clothing Image Classifier")
st.write("Upload an outfit/clothing image and the model will predict its category.")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

classes = {0 : 'Blazer', 1: 'Blouse', 2 : 'Body', 3 : 'Dress', 4: 'Hat', 5: 'Hoodie', 6: 'Longsleeve', 7: 'Outwear', 8: 'Pants', 9: 'Polo', 10: 'Shirt', 11: 'Shoes', 12: 'Shorts', 13: 'Skirt', 14: 'T-Shirt', 15: 'Top', 16: 'Undershirt'}

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # Preprocess image
    # -----------------------------

    # Resize to model input size
    image_resized = image.resize((256, 256))

    # Convert image to NumPy array
    image_array = np.array(image_resized)

    # Convert pixel values to float
    image_array = image_array.astype(np.float32)

    # Normalize pixels
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("Predict"):

        prediction = model.predict(image_array)

        # Get predicted class
        predicted_class = classes[np.argmax(prediction[0])]

        # Get confidence
        confidence = np.max(prediction[0]) * 100

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")