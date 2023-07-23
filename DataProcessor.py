# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 05:13:29 2023

@author: Huzaifah-Admin
"""                                                                            
                                                                               
import pandas as pd
import pickle
from datetime import date, datetime
import pytz


# Load the dictionary from the file
with open('symbol_encoding.pkl', 'rb') as file:
    symbol_encoding = pickle.load(file)

class DataProcessor():
    def __init__(self, realtime=False):
        
        if not realtime:
            self.data=pd.read_csv("temp_data.csv")
            
        elif realtime:
            self.data=None
            
            
    def Calculate_percentage_change(self,data, open_prices, assets):
        
        for symbol in assets:
            row=data[data['Symbol'] == symbol]
            open_val=row['Open'].values[0]
            
            change=((open_val-open_prices[symbol])/open_prices[symbol])*100
        
            data.loc[data['Symbol'] == symbol, 'Percentage Change'] = change
        
        data['Percentage Change'] = data['Percentage Change'].astype(float)
        
        return data
        


    def Calculate_top_Gainers(self,data):
        data['Top_Gainer'] = False
        grouped_data = data.groupby('Hit Time')
        for _, group in grouped_data:
            top_gainers = group[group['Percentage Change'] > 0].nlargest(10, 'Percentage Change').index
            data.loc[top_gainers, 'Top_Gainer'] = True
             
        return data
             
             
    def Calculate_top_Losers(self,data):
        data['Top_Loser']= False
        grouped_data = data.groupby('Hit Time')
        for _, group in grouped_data:
            top_losers = group[group["Percentage Change"]<0].nsmallest(10, 'Percentage Change').index
            data.loc[top_losers, 'Top_Loser'] = True
            
        return data
    
    def Encode_symbols(self,data):
        data['Symbol'] = data['Symbol'].map(symbol_encoding)
        data["time"]=data['Hit Time']
        data=data.drop(columns=['Hit Time'])
        data['Symbol_Decoded'] = data['Symbol'].map({v: k for k, v in symbol_encoding.items()})
        data=data.sort_values(by='time',ascending=True)
        all_symbols = list(symbol_encoding.keys()) 
        all_minutes = data['time'].unique()
        all_combinations = pd.MultiIndex.from_product([all_minutes, all_symbols], names=['time', 'Symbol'])
        all_data = pd.DataFrame(index=all_combinations).reset_index()
        
        return data














    




