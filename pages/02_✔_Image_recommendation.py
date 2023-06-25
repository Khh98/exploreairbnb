import streamlit as st
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

import requests
import plotly.express as px
import numpy as np 
import pandas as pd 
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import urllib.parse



import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import pairwise_distances

import tensorflow as tf
from keras.preprocessing import image
from keras.layers import Input
from keras.backend import reshape
from keras.utils import load_img,img_to_array,array_to_img
from sklearn.neighbors import NearestNeighbors
import cv2

import os

import re

#load listings data of 7 cities

#load listings data of 7 cities

listings2 = pd.read_csv("listings2.csv")



# clean the price column
conv_price = lambda x: str(x)[1:].replace(',', '') if '$' in str(x) else x

listings2['price'] = listings2['price'].apply(conv_price)
listings2['price'] = listings2['price'].apply(lambda x: float(x))





def loadImages(path):
    '''Put files into lists and return them as one list with all images 
     in the folder'''
    image_files = sorted([os.path.join(path, file)
                          for file in os.listdir(path)])
    return image_files




#extract images from url column data to match it with the images loaded
import urllib.parse

urls = list(listings2['picture_url'])
extract_images = [urllib.parse.urlparse(i)[2].rpartition('/')[2] for i in urls]

#add column
listings2['images'] = extract_images

df2 = pd.read_csv("paths.csv")
df2.drop("paths",axis=1,inplace=True)

images_data=df2.merge(listings2, how='left')
#model

def get_embedding(model, img_name):
    # Reshape
    img = load_img(img_name, target_size=(img_width, img_height))
    # img to Array
    x   = img_to_array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)





# Pre-Trained Model
# Input Shape
img_width, img_height, _ = 224, 224, 3 

base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape = (img_width, img_height, 3))
base_model.trainable = False

# Add Layer Embedding
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

df_embs = pd.read_csv("embeddings2.csv")
# Calcule DIstance Matrix
cosine_sim = 1-pairwise_distances(df_embs, metric='cosine')

indices = pd.Series(range(len(images_data)), index=images_data.index)
def get_recommender(idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim



# Define a function to load an image from URL
def load_image_from_url(url, resized_fac=0.1):
    try:
        response = requests.get(url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None or np.mean(img) < 1:
            return None
        
        w, h, _ = img.shape
        resized = cv2.resize(img, (int(h*resized_fac), int(w*resized_fac)), interpolation=cv2.INTER_AREA)
        return resized
    
    except Exception as e:
        print(f"Error loading image from URL: {url}")
        print(e)
        return None


def plot_figures(figures, nrows = 1, ncols=1,figsize=(8, 8)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


warning_message = "⚠️ Please note that the images are loaded in real-time from Airbnb listings, and some of them may be corrupted. If you encounter any errors or see black images, please click 'Get Listing' again."
idx_ref = np.random.randint(1, images_data.shape[0], 1)[0]
get_item = st.sidebar.button('Get Listing')
        
if get_item:
    st.markdown(images_data["description"].loc[idx_ref], unsafe_allow_html=True)
    st.markdown(images_data["listing_url"].loc[idx_ref], unsafe_allow_html=True)
    # Recommendations
    idx_rec, idx_sim = get_recommender(idx_ref, images_data, top_n=6)

    # Plot reference image
    ref_image = load_image_from_url(images_data.iloc[idx_ref]['picture_url'])
    st.sidebar.image(ref_image, width=350)
    st.sidebar.caption("Check the details of the following property here with similar listings👉")
    # Display warning message
    st.sidebar.warning(warning_message)

    # Plot similar images
    figures = {str(i): load_image_from_url(row['picture_url']) for i, row in images_data.loc[idx_rec].iterrows()}
    title = "<p style='text-align:left;color:DarkRed; font-size:20px;'>Compared to similar listings:</p>"
    st.markdown(title, unsafe_allow_html=True)
    st.pyplot(plot_figures(figures, 3, 2))
    title2 = "<p style='text-align:left;color:DarkRed; font-size:20px;'>For more info, visit us here:</p>"
    st.markdown(title2, unsafe_allow_html=True)
    st.markdown(str(images_data["listing_url"].loc[idx_rec]), unsafe_allow_html=True)
