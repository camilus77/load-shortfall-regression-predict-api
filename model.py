"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import json

#Added these functions here because you cannot add a function inside a function
#i.e pre-process function
def Time_of_day(HourID):
    if HourID<=5:
        return 1
    elif HourID>=6 and HourID<=12:
        return 2
    elif HourID>=13 and HourID<=15:
        return 3
    elif HourID>=16 and HourID<=19:
        return 4
    else:
        return 5

def Season(monthID):
    if monthID>=6 and monthID<=8:
        return 1
    elif monthID>=9 and monthID<=11:
        return 2
    elif monthID==12 or monthID<=2:
        return 3
    else:
        return 4


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    df_test=feature_vector_df
    
    df_test= df_test.drop(['Unnamed: 0'], axis = 1)
    df_test['Valencia_wind_deg'] = df_test['Valencia_wind_deg'].str.extract('(\d+)')
    df_test['Valencia_wind_deg'] = pd.to_numeric(df_test['Valencia_wind_deg'])
    df_test['Seville_pressure']=df_test['Seville_pressure'].str.extract('(\d+)')
    df_test['Seville_pressure']=pd.to_numeric(df_test['Seville_pressure'])
    df_test['time']=pd.to_datetime(df_test['time'])
    df_test['year'] = df_test['time'].dt.year
    df_test['month'] = df_test['time'].dt.month
    df_test['day'] = df_test['time'].dt.day
    df_test['hour'] = df_test['time'].dt.hour
    df_test['week_day'] = df_test['time'].dt.dayofweek
    df_test['Season']=df_test['month'].apply(lambda x:Season(x))
    df_test['Time_of_day']=df_test['hour'].apply(lambda x:Time_of_day(x))
    df_test=df_test.drop(['time','month','hour'], axis=1)
    df_test['Wind_deg'] = df_test[['Valencia_wind_deg', 'Bilbao_wind_deg', 'Barcelona_wind_deg']].mean(axis=1)
    df_test['Pressure'] = df_test[['Madrid_pressure', 'Valencia_pressure', 'Bilbao_pressure', 'Barcelona_pressure']].mean(axis=1)
    df_test = df_test.drop(['Valencia_wind_deg', 'Bilbao_wind_deg', 'Barcelona_wind_deg', 'Madrid_pressure', 
              'Valencia_pressure', 'Bilbao_pressure', 'Seville_pressure', 'Barcelona_pressure'], axis =1)
    df_test=df_test.drop(['Barcelona_temp', 'Barcelona_temp_max','Madrid_temp', 'Madrid_temp_max','Seville_temp', 'Seville_temp_max','Bilbao_temp', 'Bilbao_temp_max','Valencia_temp', 'Valencia_temp_max'], axis=1)
    X_scaled = StandardScaler().fit_transform(df_test )
    df_test = pd.DataFrame(X_scaled,columns=df_test.columns)
    predict_vector=df_test
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
