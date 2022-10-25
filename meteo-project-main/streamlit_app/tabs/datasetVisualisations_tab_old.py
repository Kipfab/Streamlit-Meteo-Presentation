import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from loc_informations import Loc_informations
import folium
from folium import plugins
import ipywidgets
import geocoder
import geopy
import numpy as np
import pandas as pd
from vega_datasets import data as vds


title = "Dataset Visualisations"
sidebar_name = "Dataset Visualisations"


def run():

    sns.set()
    
    st.title(title)

    st.markdown(
        """
        Weekly representation of the percentage of days with precipitation:
        """
    )
    path = '/Users/Fabien/MeteoProject/meteo-project-main/data/weatherAUS.csv'
    data = pd.read_csv(path, sep = ',')

    #Column: Date
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    #Column: Date columns transform
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    #Column: year - Week number
    #Columns for ploting
    data['Week_Number'] = data['Date'].dt.isocalendar().week
    data['Day_Number'] = data['Date'].dt.strftime('%j').astype('int')
    data['Year_Week_Number'] = data['Date'].dt.strftime('%Y-%V')
    data['RainToday'] = data['RainToday'].replace({'No': 0, 'Yes': 1})
    data['RainTomorrow'] = data['RainTomorrow'].replace({'No': 0, 'Yes': 1})
    
    RainToday = data.groupby(by='Week_Number').count()['RainToday']
    RainToday = pd.DataFrame(RainToday)
    RainToday = RainToday.rename(columns={'RainToday': 'RainTodayCount'})

    df_temp = data.groupby(by='Week_Number').sum()['RainToday']
    df_temp = pd.DataFrame(df_temp)
    df_temp = df_temp.rename(columns={'RainToday': 'RainTodaySum'})

    RainToday = pd.merge(RainToday, df_temp, left_index=True, right_index=True)

    RainToday['DaywithRain_per_NumberOfObservation'] = RainToday['RainTodaySum']/RainToday['RainTodayCount']*100
    
    st.write(RainToday)
    
    #Dataframe "Rainfall_by_week" totals precipitation in mm for individual locations year by year, week by week
    Rainfall_by_week = data.groupby(by=['Location','Week_Number','Year_Week_Number']).sum()['Rainfall']
    Rainfall_by_week = pd.DataFrame(Rainfall_by_week)
    Rainfall_by_week = Rainfall_by_week.rename(columns={'RainToday': 'Rainfall'})
    Rainfall_by_week = Rainfall_by_week.reset_index()
    Rainfall_by_week = Rainfall_by_week[Rainfall_by_week['Rainfall'] > 0]
    
    st.write(Rainfall_by_week)
    
    #PLOT    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(3, 1, 1)
    
    sns.barplot(data=RainToday, x=RainToday.index, y='DaywithRain_per_NumberOfObservation', ax=ax);
    plt.xlabel('');
    plt.ylabel('Percentage of days with precipitation');
    #Data Labels
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=12, color='black', ha='center', va='bottom')


    ax = fig.add_subplot(3, 1, 2)
    sns.boxplot(data=Rainfall_by_week, x='Week_Number', y='Rainfall', ax=ax, flierprops = dict(markerfacecolor = '0.50', markersize = 2)); 
    # flierprops = dict(markerfacecolor = '0.50', markersize = 2) - to change the size of the outliers markers to make them less distracting for people who look at the chart.
    plt.ylim(0, 500);
    plt.xlabel('');
    plt.ylabel('Rainfall [mm]');

    ax = fig.add_subplot(3, 1, 3)
    sns.boxplot(data=Rainfall_by_week, x='Week_Number', y='Rainfall', ax=ax, showfliers = False); 
    #showfliers = False - To remove the outliers from the chart, I have to specify the “showfliers” parameter and set it to false.
    plt.xlabel('Week Number');
    plt.ylabel('Rainfall [mm]');
    
    st.markdown(
        """
        FOLIUM
        """
    )
    
    path = '/Users/Fabien/MeteoProject/meteo-project-main/data/df_fav.csv'
    df_fav = pd.read_csv(path, sep = ';')
    # Deleting rows containing outliers
    df_fav = df_fav.drop(df_fav.loc[df_fav['Rainfall'] > 2].index)
    df_fav = df_fav.drop(df_fav.loc[df_fav['WindSpeed9am'] > 37].index)
    
    def getMarkerColor(minValue, maxValue, meanValue, currentValue):
        """
        The getMarkerColor function aims at defining the color associated to each australian location (city).

        Parameters:
        ----------
        minValue: the australian minimum value of the studied feature 
        maxValue: the australian maximum value of the studied feature 
        meanValue: the australian average value of the studied feature 
        currentValue: the city average value of the studied feature 
    
        Returns:
        blue: if the currentValue belongs to very_low_interval (notion of far below from the average)
        green: if the currentValue belongs to low_interval (notion of below from the average)
        orange: if the currentValue belongs to high_interval (notion of upper from the average)
        red: if the currentValue belongs to very_high_interval (notion of far upper from the average)
    
        """    
        very_low_interval = [minValue, meanValue - (meanValue - minValue)/2]
        low_interval = [meanValue - (meanValue - minValue)/2, meanValue]
        high_interval = [meanValue, meanValue + (maxValue - meanValue)/2]
        very_high_interval = [meanValue + (maxValue - meanValue)/2, maxValue]
    
        if (currentValue >= very_low_interval[0]) & (currentValue < very_low_interval[1]):
            return 'blue'
        if (currentValue >= low_interval[0]) & (currentValue < low_interval[1]):
            return 'green'
        if (currentValue >= high_interval[0]) & (currentValue < high_interval[1]):
            return 'orange'
        return 'red' 
        
    def displayMap(feature_name, season = 'None'):
        """
        The displayHeatMap function aims at displaying the HeatMap for cities according to a specific feature and
        a specific season over the dataset.
        If no season is specified, the HeatMap is displayed for the whole years over the dataset.

        Parameters:
        ----------
        feature_name: specific feature to be displayed (can be Rainfall, Sunshine, WindSpeed9am, Cloud9am)
        season: the seasion to be considered

        """
        # print('feature_name=', feature_name, ', season=', season)
        loc_informations_List = []
        # Create instances of loc_informations class and fill loc_informations_List 
        locations = df_fav['Location'].unique()
        # print(locations)
        for location in locations:
            latitude = df_fav.loc[df_fav['Location'] == location]['Latitude'].unique()
            longitude = df_fav.loc[df_fav['Location'] == location]['Longitude'].unique()
            loc_informations = Loc_informations(name = location, latitude = latitude, longitude = longitude)
            loc_informations_List.append(loc_informations)
    
        variable_names = ['Rainfall', 'Sunshine', 'WindSpeed9am', 'Cloud9am']
        if (season == 'None'):
            # Update variable_name mean for each instance contained in loc_informations_List for all season
            for loc_informations in loc_informations_List:
                for variable_name in variable_names:
                    mean = df_fav.loc[df_fav['Location'] == loc_informations.name][variable_name].mean()
                    # print('location=', loc_informations.name, ', variable_name=', variable_name, ', mean=', round(mean, 2))
                    if (variable_name == 'Rainfall'):
                        loc_informations.set_Rainfall_Mean(round(mean, 2))
                    if (variable_name == 'Sunshine'):
                        loc_informations.set_Sunshine_Mean(round(mean, 2))
                    if (variable_name == 'WindSpeed9am'):
                        loc_informations.set_WindSpeed9am_Mean(round(mean, 2))
                    if (variable_name == 'Cloud9am'):
                        loc_informations.set_Cloud9am_Mean(round(mean, 2))
    
        if (season != 'None'):
            # Update variable_name mean for each instance contained in loc_informations_List for the given season
            for loc_informations in loc_informations_List:
                for variable_name in variable_names:
                    mean = df_fav.loc[(df_fav['Location'] == loc_informations.name) & (df_fav['Season'] == season)][variable_name].mean()
                    if (variable_name == 'Rainfall'):
                        loc_informations.set_Rainfall_Mean(round(mean, 2))
                    if (variable_name == 'Sunshine'):
                        loc_informations.set_Sunshine_Mean(round(mean, 2))
                    if (variable_name == 'WindSpeed9am'):
                        loc_informations.set_WindSpeed9am_Mean(round(mean, 2))
                    if (variable_name == 'Cloud9am'):
                        loc_informations.set_Cloud9am_Mean(round(mean, 2))
        
        # Creating Australian map
        map_cities = folium.Map(location = [-25, 135], zoom_start = 4)
   
        heatmap_data = []
        scaleFactor = 10000
        marker_number = 0
        for loc_informations in loc_informations_List:
            # Checking Nan Values
            if (str(loc_informations.name) == str(np.nan)):
                # print('Nan Values when reading loc_informations.name for location', loc_informations.name, '==> continue!')                
                continue
            if (str(loc_informations.latitude) == str(np.nan)):
                # print('Nan Values when reading loc_informations.latitude for location', loc_informations.name, '==> continue!')                
                continue
            if (str(loc_informations.longitude) == str(np.nan)):
                # print('Nan Values when reading loc_informations.longitude for location', loc_informations.name, '==> continue!')                
                continue
            if (feature_name == 'Rainfall'):
                if (str(loc_informations.rainfall_mean) == str(np.nan)):
                    # print('Nan Values contained in variable Rainfall for location', loc_informations.name, '==> continue!')                
                    continue
            if (feature_name == 'Sunshine'):
                if (str(loc_informations.sunshine_mean) == str(np.nan)):
                    # print('Nan Values contained in variable Sunshine for location', loc_informations.name, '==> continue!')
                    continue
            if (feature_name == 'WindSpeed9am'):
                if (str(loc_informations.windspeed9am_mean) == str(np.nan)):
                    # print('Nan Values contained in variable WindSpeed9am for location', loc_informations.name, '==> continue!')
                    continue 
            if (feature_name == 'Cloud9am'):
                if (str(loc_informations.cloud9am_mean) == str(np.nan)):
                    # print('Nan Values contained in variable Cloud9am for location', loc_informations.name, '==> continue!')
                    continue                 
        
            # All is OK ==> Ploting locations
            # Australian map using markers to point out cities contained in the dataset
            tooltip = loc_informations.name
            popup = 'Rainfall: ' + str(loc_informations.rainfall_mean) + ' mm' + '\n'     
            popup += 'Sunshine: ' + str(loc_informations.sunshine_mean) + ' h' + '\n'
            popup += 'WindSpeed9am: ' + str(loc_informations.windspeed9am_mean) + ' km/h' + '\n'
            popup += 'Cloud9am: ' + str(loc_informations.cloud9am_mean) + ' oktas'
        
            # Building Marker
            # Getting Marker color
            if (feature_name == 'Rainfall'):
                currentValue = loc_informations.rainfall_mean
            if (feature_name == 'Sunshine'):
                currentValue = loc_informations.sunshine_mean
            if (feature_name == 'WindSpeed9am'):
                currentValue = loc_informations.windspeed9am_mean
            if (feature_name == 'Cloud9am'):
                currentValue = loc_informations.cloud9am_mean
            
            if (season == 'None'):
                minValue = df_fav[feature_name].min()
                maxValue = df_fav[feature_name].max()
                meanValue = df_fav[feature_name].mean()
                color = getMarkerColor(minValue, maxValue, meanValue, currentValue)
                # print('location=', loc_informations.name, ', color=', color)
            
            if (season != 'None'):
                minValue = df_fav.loc[df_fav['Season'] == season][feature_name].min()
                maxValue = df_fav.loc[df_fav['Season'] == season][feature_name].max()
                meanValue = df_fav.loc[df_fav['Season'] == season][feature_name].mean()
                color = getMarkerColor(minValue, maxValue, meanValue, currentValue)
                # print('location=', loc_informations.name, ', color=', color)

            marker = folium.Marker(location = [loc_informations.latitude, loc_informations.longitude], tooltip = tooltip, popup = popup, icon = folium.Icon(color = color, icon_color = color, icon = ''))
            marker.add_to(map_cities)
            marker_number += 1
            
    
        print('Ploting', marker_number, 'locations.')
    
        # Add full screen button to map
        plugins.Fullscreen(position='topright').add_to(map_cities)

        return map_cities
        
    displayMap('Rainfall')
