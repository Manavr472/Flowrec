import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define class labels
class_labels = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulips"]

# Streamlit UI
st.title("Flowrec: Flower Recognition")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    image = image.resize((224, 224))  # Resize to match the input size of your model
    image = np.array(image)
    image = image / 255.0  # Normalize the image

    
    # Make predictions
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    if st.button('Show Prediction'):

        # Get the predicted class
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the results
        st.subheader(f"Predicted Class: {predicted_class}")
        
        st.subheader("Class Probabilities:")
        probability_table = {
            "Class": class_labels,
            "Probability": predictions[0]
        }
        st.table(probability_table)

