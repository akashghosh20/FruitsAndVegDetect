# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
#
# # Load the TensorFlow Lite model
# model_path = "vegModels.tflite"
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Streamlit UI
# st.title('Fruit and Vegetable Classifier')
# st.write('Upload an image and let me classify it.')
#
# uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
#
# if uploaded_image is not None:
#     # Preprocess the image to match the model's input shape
#     image = Image.open(uploaded_image)
#     image = image.resize((224, 224))  # Resize to match model's input size
#     image = np.array(image)
#     image = (image / 255.0).astype(np.float32)  # Normalize to [0, 1]
#     image = image[np.newaxis, ...]  # Add batch dimension
#
#     # Run inference
#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details[0]['index'])
#
#     # Get the predicted class
#     predicted_class_index = np.argmax(output)
#
#     # Display the result
#     st.image(image[0], caption='Uploaded Image', use_column_width=True)
#     st.write(f'Predicted Class Index: {predicted_class_index}')
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TensorFlow Lite model
model_path = "vegModels.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels from the text file
with open('labelsvegetables.txt', 'r') as file:
    class_labels = file.read().splitlines()

# Streamlit UI
st.title('Fruit and Vegetable Classifier')
st.write('Upload an image and let me classify it.')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Preprocess the image to match the model's input shape
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # Resize to match model's input size
    image = np.array(image)
    image = (image / 255.0).astype(np.float32)  # Normalize to [0, 1]
    image = image[np.newaxis, ...]  # Add batch dimension

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class index
    predicted_class_index = np.argmax(output)

    # Get the predicted class label name
    predicted_label = class_labels[predicted_class_index]

    # Display the result
    st.image(image[0], caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Class: {predicted_label}')
