#!/usr/bin/env python
# coding: utf-8

# In[8]:

import tensorflow
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np


# In[9]:


model = load_model('xception_v6_1_12_0.834.h5')


# In[10]:


def predict_image(image):
    img = image.resize((299, 299))
    x = np.array(img)
    X = np.array([x])
    X.shape

    X = preprocess_input(X)
    pred = model.predict(X)
    result = list(zip(class_names, pred[0]))
    max_result = max(result, key=lambda x:x[1])

    return max_result


# In[11]:


st.title("Sea Animal Classification")
st.write("Upload an image of a sea animal and the model will predict its class.")


# In[12]:


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# In[13]:


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
       
    class_names = [  'Clams',
                     'Corals',
                     'Crabs',
                     'Dolphin',
                     'Eel',
                     'Fish',
                     'Jelly Fish',
                     'Lobster',
                     'Nudibranchs',
                     'Octopus',
                     'Otter',
                     'Penguin',
                     'Puffers',
                     'Sea Rays',
                     'Sea Urchins',
                     'Seahorse',
                     'Seal',
                     'Sharks',
                     'Shrimp',
                     'Squid',
                     'Starfish',
                     'Turtle_Tortoise',
                     'Whale']  # Replace with your actual class names
    label, confidence = predict_image(image)
    st.write(f"Prediction: {class_names[label]}")
    st.write(f"Confidence: {confidence:.2f}")


# In[ ]:





# In[ ]:




