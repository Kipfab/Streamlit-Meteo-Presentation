import streamlit as st
from PIL import Image


title = "Forecasting PreciPytation by Classification Models"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    
    imagesPathRoot = 'meteo-project-main/images/'

    st.title(title)

    img = Image.open(imagesPathRoot + "Introduction.png")
    st.image(img, width = 700)
    
    st.header('Objective')
    st.markdown(
        """          
        • Predict whether it will rain or now on the next day  
        • Method: Machine Learning  
	    """
    )
