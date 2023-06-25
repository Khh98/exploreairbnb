import streamlit as st
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

import requests
import numpy as np 
import pandas as pd 
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo


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
import cv2



import os

import re

import requests
from streamlit_lottie import st_lottie

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_5i5tlydx.json")



#Homepage configuration
# Fetch the image from the GitHub repository
im = Image.open("/exploreairbnb/airbnb-2.jpg")
st.set_page_config(
    page_title="Airbnb",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded")

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
header{visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ---Side-Bar----

with st.sidebar:
    st_lottie(lottie_coding, height=200, key="coding")
    st.title("Analyzing users features for a smoother travel experience")
    st.write('''
    Hey 👋🏻there. My name is Karim Hazimeh, aspiring data analyst. 
    I enjoy working with data and extracting insightful information from them🎯.
    ''')
    st.write("---")



#load listings data of 7 cities

listings2 = pd.read_csv("listings2.csv")



# clean the price column
conv_price = lambda x: str(x)[1:].replace(',', '') if '$' in str(x) else x

listings2['price'] = listings2['price'].apply(conv_price)
listings2['price'] = listings2['price'].apply(lambda x: float(x))




 #data exploration
title3 = "<p style='text-align:center;color:red; font-size:30px;'>Data Exploration</p>" 
st.markdown(title3, unsafe_allow_html=True)
matplotlib.use("Agg")
fig, ax = plt.subplots()
matplotlib.rcParams.update({"font.size": 8})
st.set_option("deprecation.showPyplotGlobalUse", False)


def categorical_column(df, max_unique_values=15):
    categorical_column_list = []
    for column in df.columns:
        if df[column].nunique() < max_unique_values:
            categorical_column_list.append(column)
    return categorical_column_list


def eda(df):

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Numbers of rows to view", 5)
        st.dataframe(df.head(number))

    # Show Columns
    if st.checkbox("Columns Names"):
        st.write(df.columns)

    # Show Shape
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
        if data_dim == "Columns":
            st.text("Numbers of Columns")
            st.write(df.shape[1])
        elif data_dim == "Rows":
            st.text("Numbers of Rows")
            st.write(df.shape[0])
        else:
            st.write(df.shape)

    # Select Columns
    if st.checkbox("Select Column to show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select Columns", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Show Value Count
    if st.checkbox("Show Value Counts"):
        all_columns = df.columns.tolist()
        selected_columns = st.selectbox("Select Column", all_columns)
        st.write(df[selected_columns].value_counts())

    # Show Datatypes
    if st.checkbox("Show Data types"):
        st.text("Data Types")
        st.write(df.dtypes)

    # Show Summary
    if st.checkbox("Show Summary"):
        st.text("Summary")
        st.write(df.describe().T)

    # Plot and visualization
    st.subheader("Data Visualization")
    all_columns_names = df.columns.tolist()

    # Correlation Seaborn Plot
    if st.checkbox("Show Correlation Plot"):
        st.success("Generating Correlation Plot ...")
        if st.checkbox("Annot the Plot"):
            st.write(sns.heatmap(df.corr(), annot=True))
        else:
            st.write(sns.heatmap(df.corr()))
        st.pyplot()

    # Count Plot
    if st.checkbox("Show Value Count Plots"):
        x = st.selectbox("Select Categorical Column", all_columns_names)
        st.success("Generating Plot ...")
        if x:
            if st.checkbox("Select Second Categorical column"):
                hue_all_column_name = df[df.columns.difference([x])].columns
                hue = st.selectbox("Select Column for Count Plot", hue_all_column_name)
                st.write(sns.countplot(x=x, hue=hue, data=df, palette="Set2"))
            else:
                st.write(sns.countplot(x=x, data=df, palette="Set2"))
            st.pyplot()

    # Pie Chart
    if st.checkbox("Show Pie Plot"):
        all_columns = categorical_column(df)
        selected_columns = st.selectbox("Select Column", all_columns)
        if selected_columns:
            st.success("Generating Pie Chart ...")
            st.write(df[selected_columns].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()
    #map
    if st.checkbox("Show Interactive map"):
        st.success("Generating Map ...")
        cscale = [
          [0.0, 'rgb(165,0,38)'], 
          [0.0005, 'rgb(215,48,39)'], 
          [0.007, 'rgb(250, 152, 122)'], 
          [0.08, 'rgb(208, 254, 144)'], 
          [0.1, 'rgb(0, 255, 179)'], 
          [0.3, 'rgb(171,217,233)'], 
          [0.7, 'rgb(116,173,209)'], 
          [0.9, 'rgb(69,117,180)'], 
          [1.0, 'rgb(49,54,149)']
         ]
        london_listings = df[df['City'] == 'London']
        barcelona_listings = df[df['City'] == 'Barcelona']
        istanbul_listings = df[df['City'] == 'Istanbul']
        ny_listings = df[df['City'] == 'New York']
        singapore_listings = df[df['City'] == 'Singapore']
        sydney_listings = df[df['City'] == 'Sydney']
        rio_listings = df[df['City'] == 'Rio de Janeiro']
        all_cities = ['London','Barcelona','Istanbul','New York','Singapore','Sydney','Rio de Janeiro']
        selected_city = st.selectbox("Select City", all_cities)
        if selected_city=="London":
                map = px.scatter_mapbox(london_listings, lat="latitude", lon="longitude",opacity=1.0, 
                        color ='price', size="price",
                        color_continuous_scale=cscale,
                        height = 900, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')

                map.update_layout(mapbox_style="open-street-map")
                map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})

                st.plotly_chart(map)
        elif selected_city=='Barcelona':
                map = px.scatter_mapbox(barcelona_listings, lat="latitude", lon="longitude",opacity=1.0, 
                        color ='price', size="price",
                        color_continuous_scale=cscale,
                        height = 900, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')

                map.update_layout(mapbox_style="open-street-map")
                map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})

                st.plotly_chart(map)
        elif selected_city=='Istanbul':
                map = px.scatter_mapbox(istanbul_listings, lat="latitude", lon="longitude",opacity=1.0, 
                        color ='price', size="price",
                        color_continuous_scale=cscale,
                        height = 900, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')

                map.update_layout(mapbox_style="open-street-map")
                map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})

                st.plotly_chart(map)
        elif selected_city=='New York':
                map = px.scatter_mapbox(ny_listings, lat="latitude", lon="longitude",opacity=1.0, 
                        color ='price', size="price",
                        color_continuous_scale=cscale,
                        height = 900, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')

                map.update_layout(mapbox_style="open-street-map")
                map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})

                st.plotly_chart(map)
        elif selected_city=='Singapore':
                map = px.scatter_mapbox(singapore_listings, lat="latitude", lon="longitude",opacity=1.0, 
                        color ='price', size="price",
                        color_continuous_scale=cscale,
                        height = 900, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')

                map.update_layout(mapbox_style="open-street-map")
                map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})

                st.plotly_chart(map)
        elif selected_city=='Sydney':
                map = px.scatter_mapbox(sydney_listings, lat="latitude", lon="longitude",opacity=1.0, 
                        color ='price', size="price",
                        color_continuous_scale=cscale,
                        height = 900, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')

                map.update_layout(mapbox_style="open-street-map")
                map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})

                st.plotly_chart(map)
        elif selected_city=='Rio de Janeiro':
                map = px.scatter_mapbox(rio_listings, lat="latitude", lon="longitude",opacity=1.0, 
                        color ='price', size="price",
                        color_continuous_scale=cscale,
                        height = 900, zoom = 9.7,
                        text= 'room_type',
                        hover_name = 'name')

                map.update_layout(mapbox_style="open-street-map")
                map.update_layout(margin={"r":80,"t":80,"l":80,"b":80})

                st.plotly_chart(map)
    
    # Customizable Plot
    st.subheader("Customizable Plot")

    type_of_plot = st.selectbox(
        "Select type of Plot", ["area", "bar", "line", "hist", "box", "kde"]
    )
    selected_columns_names = st.multiselect("Select Columns to plot", all_columns_names)

    if st.button("Generate Plot"):
        st.success(
            "Generating Customizable Plot of {} for {}".format(
                type_of_plot, selected_columns_names
            )
        )

        custom_data = df[selected_columns_names]
        if type_of_plot == "area":
            st.area_chart(custom_data)

        elif type_of_plot == "bar":
            st.bar_chart(custom_data)

        elif type_of_plot == "line":
            st.line_chart(custom_data)

        elif type_of_plot:
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
unnecessary_columns= ['scrape_id', 'last_scraped', 'source', 'host_id','host_url','host_thumbnail_url','host_picture_url','host_neighbourhood','host_listings_count',
'host_total_listings_count','host_verifications','neighbourhood_cleansed','neighbourhood_group_cleansed' ,'bathrooms',
'minimum_minimum_nights','maximum_minimum_nights', 'minimum_maximum_nights','maximum_maximum_nights', 'minimum_nights_avg_ntm',
'maximum_nights_avg_ntm', 'calendar_updated', 'calendar_last_scraped','number_of_reviews_ltm', 'number_of_reviews_l30d','license',
'calculated_host_listings_count','calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms',
'calculated_host_listings_count_shared_rooms', 'reviews_per_month','availability_30', 'availability_60', 'availability_90']
explore = listings2.drop(unnecessary_columns,axis=1).dropna()
eda(explore)


