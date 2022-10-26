# import streamlit as st
# from asyncio.windows_events import NULL
from operator import concat
import pandas as pd
import numpy as np
from PIL import Image
import os.path as path
from joblib import dump, load
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from imblearn.metrics import classification_report_imbalanced

title = "Live Predictions"
sidebar_name = "Live Predictions"

def pprreedd(location = 'Perth'):

    test_location = location
    pathl = location + '_09_2022.csv'

    data_path =  path.abspath(path.join(__file__ ,"../../.."))
    data_path = data_path + "/data/" + pathl
    data_path = data_path.replace("/", "\\")

    df_test = pd.read_csv(data_path, encoding='latin', skiprows=6)
    df_test.drop('Unnamed: 0', axis=1, inplace=True)

    df_test['Location'] = test_location

    data_climate =  path.abspath(path.join(__file__ ,"../../.."))
    data_climate = data_climate + "/data/" + 'Location_climate.csv'
    data_climate = data_climate.replace("/", "\\")

    df_climate = pd.read_csv(data_climate, sep=';')
    df_climate.drop(['Koppen climate classification desc'], axis=1, inplace=True)

    df_test = pd.merge(left=df_test, right=df_climate)

    df_test.columns = ['Date', 'MinTemp', 'MaxTemp','Rainfall', 'Evaporation', 'Sunshine','WindGustDir', 'WindGustSpeed','Time of maximum wind gust', 'Temp9am','Humidity9am', 'Cloud9am','WindDir9am', 'WindSpeed9am', 'Pressure9am','Temp3pm', 'Humidity3pm','Cloud3pm', 'WindDir3pm','WindSpeed3pm', 'Pressure3pm', 'Location','Koppen climate classification']

    df_test.drop('Time of maximum wind gust', axis=1, inplace=True)
    df_test['Date'] = pd.to_datetime(df_test['Date'], format='%Y-%m-%d')

    df_test['Year'] = df_test['Date'].dt.year
    df_test['Month'] = df_test['Date'].dt.month
    df_test['Day'] = df_test['Date'].dt.day

    df_test['Week_Number'] = df_test['Date'].dt.isocalendar().week
    df_test['Day_Number'] = df_test['Date'].dt.strftime('%j').astype('int')
    df_test['Year_Week_Number'] = df_test['Date'].dt.strftime('%Y-%V')

    df_test['RainToday'] = 0
    df_test['RainToday'] = df_test.apply(lambda x : 1 if x.Rainfall >= 1.0 else 0, axis=1)

    df_prediction = df_test

    df_test = df_test.drop('Date', axis=1)
    df_test = df_test.drop('Year_Week_Number', axis=1)
    df_test = df_test.drop('Year', axis=1)
    df_test = df_test.drop('Day', axis=1)

    df_4 = df_test
    df_4 = df_4.drop('Location', axis=1)

    # Koppen climate classification
    # csa
    if df_4['Koppen climate classification'][0] == 'csa':
        df_4['climate_clas_aw'] = 0
        df_4['climate_clas_bsk'] = 0
        df_4['climate_clas_bwh'] = 0
        df_4['climate_clas_cfa'] = 0
        df_4['climate_clas_cfb'] = 0
        df_4['climate_clas_csa'] = 1
        df_4['climate_clas_csb'] = 0
    # cfa
    if df_4['Koppen climate classification'][0] == 'cfa':
        df_4['climate_clas_aw'] = 0
        df_4['climate_clas_bsk'] = 0
        df_4['climate_clas_bwh'] = 0
        df_4['climate_clas_cfa'] = 1
        df_4['climate_clas_cfb'] = 0
        df_4['climate_clas_csa'] = 0
        df_4['climate_clas_csb'] = 0

    # bwh
    if df_4['Koppen climate classification'][0] == 'bwh':
        df_4['climate_clas_aw'] = 0
        df_4['climate_clas_bsk'] = 0
        df_4['climate_clas_bwh'] = 1
        df_4['climate_clas_cfa'] = 0
        df_4['climate_clas_cfb'] = 0
        df_4['climate_clas_csa'] = 0
        df_4['climate_clas_csb'] = 0

    # bsk
    if df_4['Koppen climate classification'][0] == 'bsk':
        df_4['climate_clas_aw'] = 0
        df_4['climate_clas_bsk'] = 1
        df_4['climate_clas_bwh'] = 0
        df_4['climate_clas_cfa'] = 0
        df_4['climate_clas_cfb'] = 0
        df_4['climate_clas_csa'] = 0
        df_4['climate_clas_csb'] = 0

    # aw
    if df_4['Koppen climate classification'][0] == 'aw':
        df_4['climate_clas_aw'] = 1
        df_4['climate_clas_bsk'] = 0
        df_4['climate_clas_bwh'] = 0
        df_4['climate_clas_cfa'] = 0
        df_4['climate_clas_cfb'] = 0
        df_4['climate_clas_csa'] = 0
        df_4['climate_clas_csb'] = 0

    # cfa / csb
    if df_4['Koppen climate classification'][0] == 'csb':
        df_4['climate_clas_aw'] = 0
        df_4['climate_clas_bsk'] = 0
        df_4['climate_clas_bwh'] = 0
        df_4['climate_clas_cfa'] = 0
        df_4['climate_clas_cfb'] = 0
        df_4['climate_clas_csa'] = 0
        df_4['climate_clas_csb'] = 0

    df_4 = df_4.drop('Koppen climate classification', axis=1)   

    # WindGustDir_N
    df_4['WindGustDir_N']=0
    df_4.loc[ (df_4['WindGustDir'] == 'N'), ['WindGustDir_N']] = '4'
    df_4.loc[ (df_4['WindGustDir'] == 'NNW') | (df_4['WindGustDir'] == 'NNE'), ['WindGustDir_N']] = '3'
    df_4.loc[ (df_4['WindGustDir'] == 'NW') | (df_4['WindGustDir'] == 'NE'), ['WindGustDir_N']] = '2'
    df_4.loc[ (df_4['WindGustDir'] == 'WNW') | (df_4['WindGustDir'] == 'ENE'), ['WindGustDir_N']] = '1'

    # WindGustDir_S
    df_4['WindGustDir_S']=0
    df_4.loc[ (df_4['WindGustDir'] == 'S'), ['WindGustDir_S']] = '4'
    df_4.loc[ (df_4['WindGustDir'] == 'SSW') | (df_4['WindGustDir'] == 'SSE'), ['WindGustDir_S']] = '3'
    df_4.loc[ (df_4['WindGustDir'] == 'SW') | (df_4['WindGustDir'] == 'SE'), ['WindGustDir_S']] = '2'
    df_4.loc[ (df_4['WindGustDir'] == 'WSW') | (df_4['WindGustDir'] == 'ESE'), ['WindGustDir_S']] = '1'

    # WindGustDir_W
    df_4['WindGustDir_W']=0
    df_4.loc[ (df_4['WindGustDir'] == 'W'), ['WindGustDir_W']] = '4'
    df_4.loc[ (df_4['WindGustDir'] == 'WNW') | (df_4['WindGustDir'] == 'WSW'), ['WindGustDir_W']] = '3'
    df_4.loc[ (df_4['WindGustDir'] == 'NW') | (df_4['WindGustDir'] == 'SW'), ['WindGustDir_W']] = '2'
    df_4.loc[ (df_4['WindGustDir'] == 'NNW') | (df_4['WindGustDir'] == 'SSW'), ['WindGustDir_W']] = '1'

    # WindGustDir_E
    df_4['WindGustDir_E']=0
    df_4.loc[ (df_4['WindGustDir'] == 'E'), ['WindGustDir_E']] = '4'
    df_4.loc[ (df_4['WindGustDir'] == 'ENE') | (df_4['WindGustDir'] == 'ESE'), ['WindGustDir_E']] = '3'
    df_4.loc[ (df_4['WindGustDir'] == 'NE') | (df_4['WindGustDir'] == 'SE'), ['WindGustDir_E']] = '2'
    df_4.loc[ (df_4['WindGustDir'] == 'NNE') | (df_4['WindGustDir'] == 'SSE'), ['WindGustDir_E']] = '1'
    df_4 = df_4.drop('WindGustDir', axis=1)

    # WindDir9am_N
    df_4['WindDir9am_N']=0
    df_4.loc[ (df_4['WindDir9am'] == 'N'), ['WindDir9am_N']] = '4'
    df_4.loc[ (df_4['WindDir9am'] == 'NNW') | (df_4['WindDir9am'] == 'NNE'), ['WindDir9am_N']] = '3'
    df_4.loc[ (df_4['WindDir9am'] == 'NW') | (df_4['WindDir9am'] == 'NE'), ['WindDir9am_N']] = '2'
    df_4.loc[ (df_4['WindDir9am'] == 'WNW') | (df_4['WindDir9am'] == 'ENE'), ['WindDir9am_N']] = '1'

    # WindDir9am_S
    df_4['WindDir9am_S']=0
    df_4.loc[ (df_4['WindDir9am'] == 'S'), ['WindDir9am_S']] = '4'
    df_4.loc[ (df_4['WindDir9am'] == 'SSW') | (df_4['WindDir9am'] == 'SSE'), ['WindDir9am_S']] = '3'
    df_4.loc[ (df_4['WindDir9am'] == 'SW') | (df_4['WindDir9am'] == 'SE'), ['WindDir9am_S']] = '2'
    df_4.loc[ (df_4['WindDir9am'] == 'WSW') | (df_4['WindDir9am'] == 'ESE'), ['WindDir9am_S']] = '1'

    # WindDir9am_W
    df_4['WindDir9am_W']=0
    df_4.loc[ (df_4['WindDir9am'] == 'W'), ['WindDir9am_W']] = '4'
    df_4.loc[ (df_4['WindDir9am'] == 'WNW') | (df_4['WindDir9am'] == 'WSW'), ['WindDir9am_W']] = '3'
    df_4.loc[ (df_4['WindDir9am'] == 'NW') | (df_4['WindDir9am'] == 'SW'), ['WindDir9am_W']] = '2'
    df_4.loc[ (df_4['WindDir9am'] == 'NNW') | (df_4['WindDir9am'] == 'SSW'), ['WindDir9am_W']] = '1'

    # WindDir9am_E
    df_4['WindDir9am_E']=0
    df_4.loc[ (df_4['WindDir9am'] == 'E'), ['WindDir9am_E']] = '4'
    df_4.loc[ (df_4['WindDir9am'] == 'ENE') | (df_4['WindDir9am'] == 'ESE'), ['WindDir9am_E']] = '3'
    df_4.loc[ (df_4['WindDir9am'] == 'NE') | (df_4['WindDir9am'] == 'SE'), ['WindDir9am_E']] = '2'
    df_4.loc[ (df_4['WindDir9am'] == 'NNE') | (df_4['WindDir9am'] == 'SSE'), ['WindDir9am_E']] = '1'
    df_4 = df_4.drop('WindDir9am', axis=1)

    # WindDir3pm_N
    df_4['WindDir3pm_N']=0
    df_4.loc[ (df_4['WindDir3pm'] == 'N'), ['WindDir3pm_N']] = '4'
    df_4.loc[ (df_4['WindDir3pm'] == 'NNW') | (df_4['WindDir3pm'] == 'NNE'), ['WindDir3pm_N']] = '3'
    df_4.loc[ (df_4['WindDir3pm'] == 'NW') | (df_4['WindDir3pm'] == 'NE'), ['WindDir3pm_N']] = '2'
    df_4.loc[ (df_4['WindDir3pm'] == 'WNW') | (df_4['WindDir3pm'] == 'ENE'), ['WindDir3pm_N']] = '1'

    # WindDir3pm_S
    df_4['WindDir3pm_S']=0
    df_4.loc[ (df_4['WindDir3pm'] == 'S'), ['WindDir3pm_S']] = '4'
    df_4.loc[ (df_4['WindDir3pm'] == 'SSW') | (df_4['WindDir3pm'] == 'SSE'), ['WindDir3pm_S']] = '3'
    df_4.loc[ (df_4['WindDir3pm'] == 'SW') | (df_4['WindDir3pm'] == 'SE'), ['WindDir3pm_S']] = '2'
    df_4.loc[ (df_4['WindDir3pm'] == 'WSW') | (df_4['WindDir3pm'] == 'ESE'), ['WindDir3pm_S']] = '1'

    # WindDir3pm_W
    df_4['WindDir3pm_W']=0
    df_4.loc[ (df_4['WindDir3pm'] == 'W'), ['WindDir3pm_W']] = '4'
    df_4.loc[ (df_4['WindDir3pm'] == 'WNW') | (df_4['WindDir3pm'] == 'WSW'), ['WindDir3pm_W']] = '3'
    df_4.loc[ (df_4['WindDir3pm'] == 'NW') | (df_4['WindDir3pm'] == 'SW'), ['WindDir3pm_W']] = '2'
    df_4.loc[ (df_4['WindDir3pm'] == 'NNW') | (df_4['WindDir3pm'] == 'SSW'), ['WindDir3pm_W']] = '1'

    # WindDir3pm_E
    df_4['WindDir3pm_E']=0
    df_4.loc[ (df_4['WindDir3pm'] == 'E'), ['WindDir3pm_E']] = '4'
    df_4.loc[ (df_4['WindDir3pm'] == 'ENE') | (df_4['WindDir3pm'] == 'ESE'), ['WindDir3pm_E']] = '3'
    df_4.loc[ (df_4['WindDir3pm'] == 'NE') | (df_4['WindDir3pm'] == 'SE'), ['WindDir3pm_E']] = '2'
    df_4.loc[ (df_4['WindDir3pm'] == 'NNE') | (df_4['WindDir3pm'] == 'SSE'), ['WindDir3pm_E']] = '1'
    df_4 = df_4.drop('WindDir3pm', axis=1)


    scaler_path =  path.abspath(path.join(__file__ ,"../../.."))
    scaler_path = scaler_path + "/model_scaler/" + 'catboost_std_scaler.bin'
    scaler_path = scaler_path.replace("/", "\\")

    scaler = load(scaler_path)
    test_data = scaler.transform(df_4)

    model_path =  path.abspath(path.join(__file__ ,"../../.."))
    model_path = model_path + "/model_scaler/" + 'catboost_model'
    model_path = model_path.replace("/", "\\")

    model_from_file = CatBoostClassifier()
    model_from_file.load_model(model_path)

    y_pred = model_from_file.predict(test_data)
    from cmath import nan
    df_prediction['RainPrediction'] = nan

    for i in range(1, len(df_prediction)):
        df_prediction.loc[i, 'RainPrediction'] = y_pred[(i-1)]
    df_prediction[['Location','Date', 'RainToday', 'RainPrediction']]

    # print(classification_report_imbalanced(df_prediction['RainToday'][1:], df_prediction['RainPrediction'][1:]))
    # print("Accuracy of predictions for location", test_location, ":", round(accuracy_score(df_prediction['RainToday'][1:], df_prediction['RainPrediction'][1:]), 2))
    return location, round(accuracy_score(df_prediction['RainToday'][1:], df_prediction['RainPrediction'][1:]), 2)

df_scores = pd.DataFrame()
df_scores['Location'] = np.nan
df_scores['Score'] = np.nan
location_list = ['Perth','GoldCoast','AliceSprings','Mildura','Townsville','Darwin','Katherine','Sydney','Cairns']

for i, locc in enumerate(location_list):
    l_name, a_score = pprreedd(locc)
    df_scores.loc[i, 'Location'] = l_name
    df_scores.loc[i, 'Score'] = a_score

df_scores.sort_values(by=["Score"], ascending=False, inplace=True)
df_scores.reset_index(inplace=True)
df_scores.drop('index', axis=1, inplace=True)

def run():

    st.title(title)

    st.markdown(
        """
        Testing the model accuracy on new samples
        Data source is: 
        Austarlias official weather forecasts & weather radar - The Bureau of Meteorology
        http://www.bom.gov.au/

        meteorological data samples are from September 2022 for selected 9 locations in Australia
        """
    )

    st.dataframe(df_scores)
    st.text( """Average accuracy = """ + str(round(df_scores['Score'].mean(), 2)) )

    # st.markdown(
    #     """
    #     opisy2 - opisy2 - opisy2
    #     """
    # )
    
    st.bar_chart(df_scores, x='Location', y='Score')

    st.markdown(
        """
        For the given data sample, the model achieved 0.83 accuracy and 0.77 F1-score.
        The model, when learned, reached an accuracy of 0.86 and an f1-score of 0.86. 
        """
    )

