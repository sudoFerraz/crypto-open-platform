from stockstats import StockDataFrame
import pandas as pd 
import numpy as np
import math

def read_Dataset(dataset_Path):
    """Receive a dataset path, reads it and returns its pandas dataframe"""
    dataset = pd.read_csv(dataset_Path)
    return dataset

def create_Indicators(dataset):
    """Receive a pandas dataframe(in given format), and return it appended with its technical indicators"""
    dataset = StockDataFrame.retype(dataset)
    macd = dataset['macdh']
    boll_lb = dataset['boll_lb']
    volume_Delta = dataset['volume_delta']
    open_2_d = bitcoin_df_indicators['open_2_d']
    stock = bitcoin_df_indicators['open_-2_r']
    cr = bitcoin_df_indicators['cr']
    sma_2 = bitcoin_df_indicators['open_2_sma']
    rsi_6 = bitcoin_df_indicators['rsi_6']
    rsi_12 = bitcoin_df_indicators['rsi_12']
    wr_10 = bitcoin_df_indicators['wr_10']
    wr_6= bitcoin_df_indicators['wr_6']
    cci = bitcoin_df_indicators['cci']
    cci_20 = bitcoin_df_indicators['cci_20']
    tr = bitcoin_df_indicators['tr']
    atr = bitcoin_df_indicators['atr']
    dma = bitcoin_df_indicators['dma']
    pdi = bitcoin_df_indicators['pdi']
    mdi = bitcoin_df_indicators['mdi']
    dx = bitcoin_df_indicators['dx']
    adx = bitcoin_df_indicators['adx']
    trix = bitcoin_df_indicators['trix']
    vr = bitcoin_df_indicators['vr']
    change_percentage = bitcoin_df_indicators['close_-1_r']
    return dataset

def cleaning_Dataset(dataset):
    """Receive a pandas dataframe, clean it, and return cleaned dataframe """
    cols = dataset.select_dtypes([np.number]).columns
    diff = dataset[cols].diff().sum()

    dataset = dataset.drop([diff==0].index, axis=1)
    dataset = dataset.drop('adj close', 1)
    dataset = dataset.fillna(method='bfill')
    dataset = dataset[1:-1]
    return dataset

def create_TargetLabel(dataset):
    """Receive a pandas dataframe, and return a new dataframe with target labels"""
    label_Array = dataset['close_-1_r'].shift(-1)
    label_Array = label_Array.apply(lambda x:1 if x>0.0000 else 0)
    return label_Array

def normalize_Features(dataset):
    """Receive a pandas dataframe and return it with its values normalized""""
    normalized_Dataset = dataset.copy()
    for feature in normalized_Dataset:
        std = normalized_Dataset[feature].std()
        mean = normalized_Dataset[feature].mean()
        normalized_Dataset[feature] = (normalized_Dataset[feature]\
            -mean)/std
        
    return normalized_Dataset

def remove_DateIndex(dataset):
    """Receive a pandas dataframe and remove the Index from it"""
    dataset.to_csv('normalized_Dataset.csv', mode='w', header=True)
    dataset = pd.read_csv("normalized_Dataset.csv")
    dataset = dataset.drop('date', axis=1)
    return dataset

def separate_TrainTest(dataset, label_Array):
    """80% train, 20%Test"""
    x_train = dataset[:((math.ceil(len(dataset)* 0.8)))]
    x_test = dataset[((math.ceil(len(dataset)* 0.8))):]
    y_train = label_Array[:((math.ceil(len(dataset)* 0.8)))]
    y_test = label_Array[((math.ceil(len(dataset)* 0.8))):]
    return x_train, x_test, y_train, y_test

