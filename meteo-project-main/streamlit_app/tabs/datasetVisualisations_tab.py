import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

title = "Dataset Visualisations"
sidebar_name = "Dataset Visualisations"


def run():

    imagesPathRoot = '../images/'
    
    st.title(title)
    
    st.header('Analysis of precipitations')
    
    st.text('Monthly representation of precipitations:')    
    img = Image.open(imagesPathRoot + "MonthlyVisualization.jpg")
    st.image(img, width = 700)

    st.text('Weekly representation of precipitations:')
    img = Image.open(imagesPathRoot + "WeeklyVisualization.jpg")
    st.image(img, width = 700)
    
        
    st.header('Koppen Climate Classification analysis')
    
    st.markdown(
        """
        Divides climate into 5 main climate groups with each group being divided again based on precipitation and temperature.  
        """
    )
    
    img = Image.open(imagesPathRoot + "KCC_Wiki.png")
    st.image(img, width = 400)
    
    st.markdown(
        """
        In Australia, there are 8 types of different climate group based on Koppen Climate Classification:  
		• Csb: Warm-summer Mediterranean climate;  
		• Am: Tropical monsoon climate;  
		• Cfb: Temperate oceanic climate;  
		• Cfa: Humid subtropical climate;  
		• Aw: Tropical wet and dry or savanna climate;  
		• Csa: Hot-summer Mediterranean climate;  
		• BSk: Cold semi-arid climate;  
		• BWk: Cold desert climate.
        """
    )
    
    st.header('Precipitation grouped by Koppen Climate Classification analysis')    
    img = Image.open(imagesPathRoot + "KCC.png")
    st.image(img, width = 1000)
    
    st.header('Precipitation grouped by Rain frequencies analysis:')
    st.markdown(
        """
        Climate zones can be arranged into four groups:        
		• Csb, Am: zones with frequent precipitation (>30%);  
		• Cfb, Cfa, Csa, Aw: zones with moderately occurring precipitation (22%);  
		• Bsk: zone with infrequent precipitation (14%);  
		• Bwh: zone with very rare precipitation (7%).  
        """
    )
    
    img = Image.open(imagesPathRoot + "RainfallFrequencyClassification.png")
    st.image(img, width = 300)
    
    
    st.header('Explore and Visualize Rainfall and Sunshine with Folium Library')    
    img = Image.open(imagesPathRoot + "Folium.png")
    st.image(img, width = 500)
    
    st.text('Rainfall and Sunshine by different locations:') 
    img = Image.open(imagesPathRoot + "Rainfall_Sunshine_Different_Locations.png")
    st.image(img, width = 700)
    
    st.text('Rainfall with Seasonal Granularities::') 
    img = Image.open(imagesPathRoot + "Rainfall_Seasonal_Granularities.png")
    st.image(img, width = 700)
    
    st.text('Sunshine with Seasonal Granularities::') 
    img = Image.open(imagesPathRoot + "Sunshine_Seasonal_Granularities.png")
    st.image(img, width = 700)
    
    

