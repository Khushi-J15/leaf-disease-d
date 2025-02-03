import streamlit as st
import numpy as np
import pickle
import gzip
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load model from compressed pickle file
def load_compressed_model(model_path):
    with gzip.open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Model prediction function
def model_prediction(test_image):
    # Load the model
    model = load_compressed_model("model_compressed.pkl.gz")  # Path to your compressed pickle model
    
    # Load and preprocess the image
    image = load_img(test_image, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Expand dimensions to match model input
    
    # Make predictions
    predictions = model.predict(input_arr)
    
    # Return the index of the highest prediction
    return np.argmax(predictions)

# Streamlit UI setup
st.sidebar.title("Plant Health Dashboard")
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Disease Recognition'])

# Logo for website 
img = Image.open('Diseases.png') 
st.image(img)

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System</h1>", unsafe_allow_html=True)

elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System for Sustainable Agriculture')

# Image uploader
test_image = st.file_uploader('Choose an image:')

if st.button('Show Image'):
    st.image(test_image, width=400, use_container_width=True)

if st.button('Predict'):
    st.snow()
    with st.spinner("Processing..."):  # Spinner animation
        st.write('Our Prediction')
        result_index = model_prediction(test_image)

        # Display the results
        class_names = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
        st.success(f'Model is predicting it\'s a {class_names[result_index]}')
