import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Dataset Modifications For Modeling"
sidebar_name = "Dataset Modifications For Modeling"


def run():

    imagesPathRoot = 'meteo-project-main/images/'
    st.title(title)

    st.header('Encoding the Location variable')
    img = Image.open(imagesPathRoot + "LocationEncoding.png")
    st.image(img, width = 1000)

    
    st.header('Encoding the Wind direction variables')    
    img = Image.open(imagesPathRoot + "WindEncoding.png")
    st.image(img, width = 750)
    
    st.header('Five datasets for modeling')
    st.markdown(
        """
		Based on the previously described ways of coding categorizing variables, five datasets are created:    
        """
    )

    img = Image.open(imagesPathRoot + "CodingCategorization.jpg")
    st.image(img, width = 750)