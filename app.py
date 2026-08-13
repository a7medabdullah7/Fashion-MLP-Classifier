import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model("fashion_mlp_model.keras")

# Labels
class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

st.title("Fashion MLP Image Classifier")

uploaded_file = st.file_uploader(
    "Upload a Fashion Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))

    st.image(image, caption="Uploaded Image", width=200)

    img = np.array(image)/255.0
    img = img.reshape(1,28,28)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    st.success(f"Prediction: {class_names[predicted_class]}")