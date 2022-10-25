import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Dataset And Features Presentation"
sidebar_name = "Dataset And Features Presentation"


def run():

    st.title(title)

    imagesPathRoot = '../images/'
    path = '../data/weatherAUS.csv'
    df_raw = pd.read_csv(path, sep = ',')
    
    st.header('Features presentation')
    img = Image.open(imagesPathRoot + "RawDataset.png")
    st.image(img, width = 1000)

    st.header('Dataset is unbalanced')	
    st.write(df_raw['RainTomorrow'].value_counts(normalize = True))