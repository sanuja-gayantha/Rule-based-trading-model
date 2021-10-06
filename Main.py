from datetime import datetime as dt
from numpy import linalg as la
from scipy.signal import argrelextrema
import os
from sklearn.svm import SVR
from itertools import chain
import math 
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import time
from datetime import datetime, timedelta
import sqlite3
import concurrent.futures
import pytz




class Currency:
    
    def __init__(self, currency_type, time_frame = mt5.TIMEFRAME_M15, numberof_bars = 501, date_from = None, date_to = None):

        self.currency_type = currency_type
        self.time_frame = time_frame
        self.numberof_bars = numberof_bars
        self.date_from = date_from 
        self.date_to = date_to


    # calculate the precentage 
    def precentChange(self, startPoint, currentPoint):
        try:
            x = ((float(currentPoint) - startPoint) / abs(startPoint))*100.00
            if x == 0.0:
                return 0.000001
            else:
                return x

        except ZeroDivisionError:
            return 0.000001 



    def get_data(self):

        # initialize mt5 object
        mt5.initialize()     

        data_values = mt5.copy_rates_from_pos(self.currency_type, self.time_frame, 0, self.numberof_bars)

        # print(self.currency_type ,len(data_values)+1, " Data Points received ")

        # shut down connection to the MetaTrader 5 terminal
        mt5.shutdown()

        #create dataframe
        data_frame = pd.DataFrame(data_values)

        # print(data_frame.iloc[-1])

        # if there is data return it
        if (data_frame is not None) and not (data_frame.empty):

            # convert time in seconds into the datetime format
            data_frame['time']=pd.to_datetime(data_frame['time'], unit='s')

            # drop unwanted columns
            data_frame.drop(['high', 'low', 'open','spread', 'real_volume'],  1, inplace=True) 

            # get the last row time / we need this time, add to database it is the trade execute time
            # ASSIGN CURRENCY CLASS RECEVING DATAFRAME LAST DROP ROW TIME TO THIS VARABLE
            remove_dateTime = data_frame.iloc[-1].time
            remove_closingPrice = data_frame.iloc[-1].close

            # drop last row
            data_frame.drop(data_frame.tail(1).index,inplace=True) # drop last n rows 

            return True
