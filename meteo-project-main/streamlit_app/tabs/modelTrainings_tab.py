import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Model Trainings"
sidebar_name = "Model Trainings"


def run():

    imagesPathRoot = 'meteo-project-main/images/'
    st.title(title)

    st.header('Metrics to evaluate models: Accuracy, Recall and F1')
    
    img = Image.open(imagesPathRoot + "Metrics.png")
    st.image(img, width = 700)
    
    
    st.header('Forecasting precipitation using KNN model')
    
    st.markdown(
        """                     
        KNN performances for each dataset and for each type of sampling method:  
        """
    )
    
    img = Image.open(imagesPathRoot + "KNN_Scores.png")
    st.image(img, width = 750)
    
    st.markdown(
        """       
        Best scores for:  
        • datasets df_4 and df_5;  
        • both with oversampling method.  
        
        KNN model handles the predictions not so well since accuracy is poor (< 80%).
        """
    )
    
    st.header('Forecasting precipitation using XGBoost model')
    
    st.markdown(
        """        
        XGBoost performances for each dataset and for each type of sampling method:  
        """
    )
    
    img = Image.open(imagesPathRoot + "XGBoost_Scores.png")
    st.image(img, width = 750)
    
    st.markdown(
        """       
        Best scores for:  
        • undersampling method;  
        • dataset independant.  
        
        XGBoost model handles the predictions better (Accuracy, Recall > 80%).
        """
    )
    
    st.header('Forecasting precipitation using Random Forest model')
    
    st.markdown(
        """     
        Random Forest performances for each dataset and for each type of sampling method:  
        """
    )
    
    img = Image.open(imagesPathRoot + "RF_Scores.png")
    st.image(img, width = 750)
    
    st.markdown(
        """       
        Best scores for:  
        • undersampling method;  
        • dataset independant.  
        
        Random Forest model handles the predictions also pretty well (Accuracy, Recall > 80%).
        """
    )
    
    st.header('Forecasting precipitation using CatBoost model')
    
    st.markdown(
        """       
        CatBoost performances for each dataset and for each type of sampling method:  
        """
    )
    
    img = Image.open(imagesPathRoot + "CatBoost_Scores.png")
    st.image(img, width = 750)
    
    st.markdown(
        """       
        Best scores for:  
        • undersampling method;  
        • dataset independant.  
        
        CatBoost model has a very good recall but poor accuracy.
        """
    )
       
    st.header('Forecasting precipitation using Logistic Regression model')
    
    st.markdown(
        """               
        Logistic Regression performances for each dataset and for each type of sampling method:  
        """
    )
    
    img = Image.open(imagesPathRoot + "LR_Scores.png")
    st.image(img, width = 750)
    
    st.markdown(
        """       
        Best scores for:  
        • over and undersampling methods;  
        • dataset independant.  
        
        Worst classifier: poor Accuracies and Recalls (< 80%).  
        """
    )
    
    st.header('Global scores')    
    st.markdown(
        """       
        All classifiers have all a relatively poor F1 scores: around 65%.  
             
        Best scores for:  
        • undersampling method.    
        
        Best classifiers:  
        • XGBoost and Random Forest give the best results and are dataset independant;   
        • Moreover, they also have good Recalls for the dominant class 0: around 80%.
        
        CatBoost is a good classifier and is also dataset independant.    
        
        Only exception for KNN classifier:  
        • best scores with df_4 and df_5 datasets and with oversampling method.  
        
        Logistic Regression is the worst classifier.  
        """
    )

