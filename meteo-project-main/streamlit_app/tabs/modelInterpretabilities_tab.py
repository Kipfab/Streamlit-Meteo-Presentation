import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os.path as path

title = "Model Interpretabilities"
sidebar_name = "Model Interpretabilities"


def run():

    imagesPathRoot = path.abspath(path.join(__file__ ,"../../.."))
    imagesPathRoot = imagesPathRoot + "/images/"
    # imagesPathRoot = imagesPathRoot.replace("/", "\\")

    # imagesPathRoot = '/Users/Fabien/MeteoProject/meteo-project-main/images/'
    st.title(title)


    st.header('KNN Model Interpretability Using Skater')
      
    st.markdown(
        """
        Model's hyperparameters:  
        • dataset: undersampled df_5;  
        • algorithm: ball tree;  
        • number of neighbors: 50.  
        
        Normalized importances of the ten most important features with KNN Model:            
        """
    )
    
    img = Image.open(imagesPathRoot + "KNNFeatureImportances.jpg")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • Features Humidity3mp, Sunshine and RainToday are among the four most important features which is rather intuitive.    
        • Feature climate_class_med is the only one resulting from the coding of categorical variables.   
        
        Cumulative feature importance with KNN model:      
        """
    )
    
    img = Image.open(imagesPathRoot + "KNNCumulativeFeatureImportance.jpg")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • to reach 95% cumulative feature importance model needs 32 features out of 35;  
        • to reach 90% cumulative feature importance model needs 28 features out of 35.        
        """
    )

    st.header('Random Forest Model Interpretability Using Skater')
      
    st.markdown(
        """
        Model's hyperparameters:  
        • dataset: undersampled df_4;    
        • criterion: “entropy”;  
        • max_features: “log2”;  
        • n_estimators = 400.  
        
        Normalized importances of the ten most important features with Random Forest Model:            
        """
    )
    
    img = Image.open(imagesPathRoot + "NormalizedImportanceRF.png")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • One variables has high importance compared to the others: Humidity3mp;  
        • No variable resulting from the coding of categorical variables;  
        
        Cumulative feature importance with Random Forest model:      
        """
    )
    
    img = Image.open(imagesPathRoot + "CumulativeFeatureImportanceRF.png")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • to reach 95% cumulative feature importance model needs 30 features out of 39;    
        • to reach 90% cumulative feature importance model needs 25 features out of 39.        
        """
    )
        
    st.header('XGBoost Model Interpretability Using Skater')
      
    st.markdown(
        """
        Model's hyperparameters:  
        • dataset: undersampled df_5;  
        • objective: binary logistic;  
        • bbase_score: 0.50;  
        • max_depth: 6.  
        
        Normalized importances of the ten most important features with XGBoost Model:            
        """
    )
    
    img = Image.open(imagesPathRoot + "XGBFeatureImportances.jpg")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • Two variables have high importance compared to the others: Humidity3mp and Pressure3pm;  
        • Feature Day_Number is the only one resulting from the coding of categorical variables;  
        • Surprising that feature Rainfall does not appear.
        
        Cumulative feature importance with XGBoost model:      
        """
    )
    
    img = Image.open(imagesPathRoot + "XGBCumulativeFeatureImportances.jpg")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • to reach 95% cumulative feature importance model needs 26 features out of 35;    
        • to reach 90% cumulative feature importance model needs 20 features out of 35.        
        """
    )
 
    
    
    st.header('Catboost Model Interpretability Using Skater')
      
    st.markdown(
        """
        Model's hyperparameters:  
        • dataset: df_4;    
        
        Normalized importances of the ten most important features with Catboost Model:            
        """
    )
    
    img = Image.open(imagesPathRoot + "NormalizedImportanceCB.png")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • Two variables has high importance compared to the others: Pressure3pm and Humidity3mp;   
        • Feature Day_Number is the only one resulting from the coding of categorical variables;  
        
        Cumulative feature importance with Catboost model:      
        """
    )
    
    img = Image.open(imagesPathRoot + "CumulativeFeatureImportanceCB.png")
    st.image(img, width = 500)
    
    st.markdown(
        """
        • to reach 95% cumulative feature importance model needs 31 features out of 39;    
        • to reach 90% cumulative feature importance model needs 25 features out of 39.        
        """
    )
    
    


