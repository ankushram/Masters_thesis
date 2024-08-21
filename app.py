import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('/content/brain_tumor_model.h5')

# Define the input size expected by the model
input_size = (224, 224)  # Model expects 224x224x3 input

# Define the class labels in the correct order
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(image):
    """
    Preprocess the uploaded image to the required input size for the model.
    """
    image = image.resize(input_size)  # Resize image to 224x224
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

st.title("Brain Tumor Classification")

# Upload an image
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make predictions
    img_array = preprocess_image(image)  # Resize and preprocess the image
    prediction = model.predict(img_array)
    
    # Decode the prediction to class name
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Display the result
    st.write(f"Prediction: {predicted_label}")
