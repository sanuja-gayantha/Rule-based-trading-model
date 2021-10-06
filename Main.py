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

            return True, data_frame, remove_dateTime, remove_closingPrice

        #else If any case data_frame becomes None
        else:

            return False, 0, 0, 0
     


    def get_price_values(self):

        mt5.initialize()

        # set time zone to UTC
        timezone = pytz.timezone("Etc/UTC") 

        # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
        utc_from = dt(self.date_from.year, self.date_from.month, self.date_from.day, 00, 00, tzinfo=timezone)
        utc_to = dt(self.date_to.year, self.date_to.month, self.date_to.day, 23, 59, tzinfo=timezone)

        # get data 
        data_values = mt5.copy_rates_range(self.currency_type, self.time_frame, utc_from, utc_to)

        # shut down connection to the MetaTrader 5 terminal
        mt5.shutdown()

        data_frame = pd.DataFrame(data_values)

        # if there is data return it
        if (data_frame is not None) and not (data_frame.empty):

            # drop unwanted columns
            data_frame.drop(['time', 'high', 'low', 'open','spread', 'real_volume'],  1, inplace=True) 

            # get the precentage of day open to current closing return it
            return round(self.precentChange(data_frame.iloc[0].close, data_frame.iloc[-1].close), 8)

        #else If any case data_frame becomes None
        else:

            return 0




class TradingModel:

    def __init__(self, df, distance_factor: float = 0.1, n: int = 5, extend_lines: bool = False, priceRangeFactor: float = 100, C = 1e3, gamma = 0.005):

        self.df = df
        self.n = n
        self.distance_factor = distance_factor
        self.extend_lines = extend_lines
        self.priceRangeFactor = priceRangeFactor
        self.C = C
        self.gamma = gamma


    def findTrends(self):

        # store all the trends information here
        trends = []

        # Find local peaks and add to dataframe
        self.df['min'] = self.df.iloc[argrelextrema(
            np.array(self.df.close.values), np.less, order=self.n)[0]]['close']
        self.df['max'] = self.df.iloc[argrelextrema(
            np.array(self.df.close.values), np.greater, order=self.n)[0]]['close']

        # self.df['min'] = self.df.iloc[argrelextrema(np.array(self.df.low.values), np.less, order=self.n)[0]]['low']
        # self.df['max'] = self.df.iloc[argrelextrema(np.array(self.df.high.values), np.greater, order=self.n)[0]]['high']

        # Extract only rows where local peaks are not null
        self.dfMax = self.df[self.df['max'].notnull()]
        self.dfMin = self.df[self.df['min'].notnull()]

        # Remove all local maximas which have other maximas close to them
        prevIndex = -1
        currentIndex = 0
        dropRows = []
        # find indices
        for i1, p1 in self.dfMax.iterrows():
            currentIndex = i1
            if currentIndex <= prevIndex + self.n * 0.64:
                dropRows.append(currentIndex)
            prevIndex = i1
        # drop them from the max self.df
        self.dfMax = self.dfMax.drop(dropRows)
        # replace with nan in initial self.df
        for ind in dropRows:
            self.df.iloc[ind, :]['max'] = np.nan

        # Remove all local minimas which have other minimas close to them
        prevIndex = -1
        currentIndex = 0
        dropRows = []
        # find indices
        for i1, p1 in self.dfMin.iterrows():
            currentIndex = i1
            if currentIndex <= prevIndex + self.n * 0.64:
                dropRows.append(currentIndex)
            prevIndex = i1
        # drop them from the min self.df
        self.dfMin = self.dfMin.drop(dropRows)
        # replace with nan in initial self.df
        for ind in dropRows:
            self.df.iloc[ind, :]['min'] = np.nan


        # Find Trends Made By Local Minimas

        """
        Itarate dataframe twise
        """

        for i1, p1 in self.dfMin.iterrows():
            for i2, p2 in self.dfMin.iterrows():
                """
                Draw trendlines based on p(i) point
                ex: p1 -> p2
                    p1 -> p3
                    ........
                """
                if i1 + 1 <= i2:
                    """
                    Check possible uptrend (starting with p1, with p2 along the way)
                    """
                    if p1['min'] < p2['min']:
                        # trendPoints list
                        trendPoints = []

                        # identify x, y points
                        p1min = p1['min']
                        p2min = p2['min']

                        p1time = i1
                        p2time = i2

                        # Put data in to asarray and create point1 & point2 for each x, y
                        point1 = np.asarray((p1time, p1min))
                        point2 = np.asarray((p2time, p2min))


                        # length of trend
                        line_length = np.sqrt(
                            (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                        # print(line_length)

                        """"
                        Now we're checking the points along the way to see how many validations happened and if the trend has ever been broken
                        Iterate all the uptrend low points between i1+1 and i2  to check line break                    
                        """
                        for i3 in range(i1 + 1, i2):
                            if not pd.isna(self.df.loc[i3, :]['min']):
                                p3 = self.df.loc[i3, :]
                                if p3['min'] < p1['min']:
                                    # if one value between the two points is smaller
                                    # than the first point, the trend has been broken
                                    trendPoints = []
                                    break

                                p3min = p3['min']
                                p3time = i3
                                point3 = np.asarray((p3time, p3min))
                                d = la.norm(
                                    np.cross(point2-point1, point1-point3))/la.norm(point2-point1)

                                v1 = (point2[0] - point1[0],
                                      point2[1] - point1[1])
                                v2 = (point3[0] - point1[0],
                                      point3[1] - point1[1])
                                xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

                                if xp < -0.0003 * self.distance_factor:
                                    trendPoints = []
                                    break

                                if d < 0.0006 * self.distance_factor:
                                    trendPoints.append({
                                        'x': i3,
                                        'y': p3["min"],
                                        'x_norm': i3,
                                        'y_norm': p3min,
                                        'dist': d,
                                        'xp': xp})

                        if len(trendPoints) > 0:
                            trends.append({
                                "direction": "up",
                                "position": "below",
                                "validations": len(trendPoints),
                                "length": line_length,
                                "x_values": [i1, i2],
                                "y_values": [p1min, p2min],
                                "i1": i1,
                                "i2": i2,
                                "p1": (i1, p1["min"]),
                                "p2": (i2, p2["min"]),
                                "color": "Green",
                                "points": trendPoints,
                                "p1_norm": (i1, p1min),
                                "p2_norm": (i2, p2min)})

          

        # # Find Trends Made By Local Maximas
        for i1, p1 in self.dfMax.iterrows():
            for i2, p2 in self.dfMax.iterrows():
                if i1 + 1 < i2:
                    if p1['max'] <= p2['max']:
                        # possible uptrend (starting with p1, with p2 along the way)
                        pass
      
                    else:
                        # possible downtrend
                        trendPoints = []

                        # normalize the starting and ending points
                        p1max = p1['max']
                        p2max = p2['max']

                        p1time = i1
                        p2time = i2

                        point1 = np.asarray((p1time, p1max))
                        point2 = np.asarray((p2time, p2max))

                        # length of trend
                        line_length = np.sqrt(
                            (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

                        # now we're checking the points along the way
                        # to see how many validations happened
                        # and if the trend has ever been broken
                        for i3 in range(i1 + 1, i2):
                            if not pd.isna(self.df.loc[i3, :]['max']):
                                p3 = self.df.loc[i3, :]

                                if p3['max'] > p1['max']:
                                    # if one value between the two points is larger
                                    # than the first point, the trend has been broken
                                    trendPoints = []
                                    break

                                # normalizing this point along the way
                                p3max = p3['max']
                                p3time = i3

                                point3 = np.asarray((p3time, p3max))

                                # distance between p3 and the line made by p1 and p2
                                d = la.norm(
                                    np.cross(point2-point1, point1-point3))/la.norm(point2-point1)

                                # cross product between p2p1 and p3p1
                                v1 = (point2[0] - point1[0],
                                      point2[1] - point1[1])
                                v2 = (point3[0] - point1[0],
                                      point3[1] - point1[1])
                                xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

                                if xp > 0.0003 * self.distance_factor:
                                    # p3 is too far above the line, therefore the trend is not valid
                                    trendPoints = []
                                    break

                                if d < 0.0006 * self.distance_factor:
                                    # p3 close enough to the line to act as a validation
                                    trendPoints.append({
                                        'x': i3,
                                        'y': p3["max"],
                                        'x_norm': p3time,
                                        'y_norm': p3max,
                                        'dist': d,
                                        'xp': xp})

                        if len(trendPoints) > 0:
                            trends.append({
                                "direction": "down",
                                "position": "above",
                                "validations": len(trendPoints),
                                "length": line_length,
                                "x_values": [i1, i2],
                                "y_values": [p1max, p2max],
                                "i1": i1,
                                "i2": i2,
                                "p1": (i1, p1["max"]),
                                "p2": (i2, p2["max"]),
                                "color": "Red",
                                "points": trendPoints,
                                "p1_norm": (p1time, p1max),
                                "p2_norm": (p2time, p2max)})

        # print("all trends :", len(trends))

        # Remove redundant trends
        removeTrends = []
        priceRange = (self.df['max'].max() /
                      self.df['min'].min()) * self.priceRangeFactor
        # print("priceRange :", priceRange)

        # Loop through trends twice
        for trend1 in trends:
            if trend1 in removeTrends:
                continue
            for trend2 in trends:
                if trend2 in removeTrends:
                    continue
                # If trends share the same starting or ending point, but not both, and the cross product
                # between their vectors is small (and so is the angle between them), remove the shortest
                if trend1["i1"] == trend2["i1"] and trend1["i2"] != trend2["i2"]:
                    v1 = (trend1["p2_norm"][0] - trend1["p1_norm"][0],
                          trend1["p2_norm"][1] - trend1["p1_norm"][1])
                    v2 = (trend2["p2_norm"][0] - trend1["p1_norm"][0],
                          trend2["p2_norm"][1] - trend1["p1_norm"][1])
                    xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

                    if xp < 0.0004 * priceRange and xp > -0.0004 * priceRange:
                        # print("p1: Trends are close to each other!")
                        # print(str(trend1['p1']) + " " + str(trend1['p2']))
                        # print(str(trend2['p1']) + " " + str(trend2['p2']))
                        if trend1['length'] > trend2['length']:
                            removeTrends.append(trend2)
                            # trends.remove(trend2)
                            trend1["validations"] = trend1["validations"] + 1
                        else:
                            removeTrends.append(trend1)
                            # trends.remove(trend1)
                            trend2["validations"] = trend2["validations"] + 1

                elif trend1["i2"] == trend2["i2"] and trend1["i1"] != trend2["i1"]:
                    v1 = (trend1["p1_norm"][0] - trend1["p2_norm"][0],
                          trend1["p1_norm"][1] - trend1["p2_norm"][1])
                    v2 = (trend2["p1_norm"][0] - trend1["p2_norm"][0],
                          trend2["p1_norm"][1] - trend1["p2_norm"][1])
                    xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

                    if xp < 0.0004 * priceRange and xp > -0.0004 * priceRange:
                        # print("p2: Trends are close to each other!")
                        # print(str(trend1['p1']) + " " + str(trend1['p2']))
                        # print(str(trend2['p1']) + " " + str(trend2['p2']))
                        if trend1['length'] > trend2['length']:
                            removeTrends.append(trend2)
                            # trends.remove(trend2)
                            trend1["validations"] = trend1["validations"] + 1
                        else:
                            removeTrends.append(trend1)
                            # trends.remove(trend1)
                            trend2["validations"] = trend2["validations"] + 1

        for trend in removeTrends:
            if trend in trends:
                trends.remove(trend)

        """
        * Identify parralel trends (above and below)
        * Get line equations based on points
        * Create lines to draw on graph
        """
        lines = []

        # Also save line equations
        lineEqs = []

        for trend in trends:
            # If trend has more than 2 validations, plot the line covering the entire chart
            if self.extend_lines and trend["validations"] > 2:

                # Find the line equation
                m = (trend["p2"][1] - trend["p1"][1]) / \
                    (trend["p2"][0] - trend["p1"][0])
                b = trend["p2"][1] - m * trend["p2"][0]
                lineEqs.append((m, b))

                # Find the last timestamp
                tMax = self.df.index.max()

                lines.append({
                    "type": "line",
                            "direction": trend["direction"],
                            "position": trend["position"],
                            "length": trend["length"],
                            "x0": trend["p1"][0],
                            "y0": trend["p1"][1],
                            "m": m,
                            "b": b,
                            "x1": tMax,
                            "x11": trend["p2"][0], #last low or high for all validation > 2 trends                           
                            "y1": m * tMax + b,
                            # "x1": trend["p2"][0],
                            # "y1": trend["p2"][1],
                            "color": trend["color"],
                            "validations": trend["validations"],
                })

            else:

                lines.append({
                    "type": "line",
                            "direction": trend["direction"],
                            "position": trend["position"],
                            "length": trend["length"],
                            "x0": trend["p1"][0],
                            "y0": trend["p1"][1],
                            "x1": trend["p2"][0],
                            "x11": trend["p2"][0],
                            "y1": trend["p2"][1],
                            "color": trend["color"],
                            "validations": trend["validations"],
                })

        # print("validate trends :", len(lines))
  

        # print(trends)

        # Concatanate Max & min
        # Max_Min = pd.concat([self.dfMax, self.dfMin]).sort_index()
        # Max_Min = Max_Min['min'].fillna(Max_Min['max'])
        # print(Max_Min)

        return lines


    def findRBFVolume(self):

        #Use scikit-learn Radial basis function to clear the noise
        X = np.array(self.df.index).reshape(-1,1)
        y = np.array(self.df['tick_volume'])
        svr_rbf = SVR(kernel='rbf', C = self.C, gamma = self.gamma)
        y_rbf = svr_rbf.fit(X, y).predict(X)     

        # r_sq = svr_rbf.score(X, y)
        # slope = (y_rbf[-1] - y_rbf [-2])
        # alpha = math.degrees(math.atan(slope))
        # print(r_sq, slope, alpha)


        #convert 2D list to 1D
        X_rbf = list(chain.from_iterable(X))
        # print(X_rbf)

        #Create RBF_Volume dataframe, so we can access data easily
        RBF_dict = {'RBF_Index': X_rbf, 'RBF_Volume': y_rbf} 
        RBF_df = pd.DataFrame(RBF_dict)
        # print(len(RBF_df))

        RBF_df.set_index('RBF_Index', inplace=True) # set RBF_Index as index

        self.df.drop(['min', 'max'], 1, inplace=True) # Drop some unwanted columns

        self.df.columns = ['time', 'close', 'real_volume'] # change tick_volume name to real_volume

        self.df['tick_volume'] = RBF_df.RBF_Volume.tolist() # add RBF volume in to self.df as tick_volume
        # print(self.df)

        return self.df






class TradeOpen:

    def __init__(self, df, lines, currency_type, remove_dateTime, VGapFactor1: float = 0.05, VGapFactor2: float = 0.15, stopLoss: float = 0.002, hGapFactor1: int = 96, hGapFactor2: int = 12, trendLen: float = 30.0, MAFactor: int = 233, positionSize: float = 0.02, takeProfit: float = None, maximumTrades: int = 5, stop_distance: int = 20):

        self.accountNumber: int = "accountNumber" #enter accountNumber as int
        self.positionSize: float = positionSize
        self.stopLoss: float = stopLoss
        self.takeProfit: float = takeProfit
        self.maximumTrades = maximumTrades
        self.stop_distance = stop_distance

        self.df = df
        self.currency_type: str = currency_type
        self.remove_dateTime = remove_dateTime
        self.lines = lines
        self.VGapFactor1 = VGapFactor1
        self.VGapFactor2 = VGapFactor2
        self.hGapFactor1 = hGapFactor1
        self.hGapFactor2 = hGapFactor2
        self.ttimedelta = timedelta(hours=3, minutes=00)
        self.trendLen = trendLen
        self.MAFactor: int = MAFactor




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
 


    # calculate VWAP
    def VWAP(self):
            
        q = self.df.tick_volume.values
        p = self.df.close.values

        VWAPdf = self.df.assign(vwap_tick_volume=(p * q).cumsum() / q.cumsum())
        VWAPdf.drop(['time', 'close', 'real_volume','tick_volume'], 1, inplace=True)  

        return VWAPdf



    # initialize conection with trading platform
    def connect(self, account):

        mt5.initialize()
        authorized=mt5.login(int(account))

        if authorized:
            print("Connected: Connecting to MT5 Client")
        else:
            print("Failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))




    # close the connection
    def disconnect(self):  

        # shut down connection to the MetaTrader 5 terminal
        mt5.shutdown()



    # this function send trade open order to trade server          
    def open_position(self, order_type, slCalPrice):

        try:

            symbol_info = mt5.symbol_info(self.currency_type)
            point = symbol_info.point * 10

            if symbol_info is None:
                print(self.currency_type, "not found")
                return 

            if not symbol_info.visible:
                # print(self.currency_type, "is not visible, trying to switch on")
                if not mt5.symbol_select(self.currency_type, True):
                    print("symbol_select({}}) failed, exit",self.currency_type)
                    return 

            # print(self.currency_type, "found!")

            
            if(order_type == "buy"):
                order = mt5.ORDER_TYPE_BUY
                currentPrice = mt5.symbol_info_tick(self.currency_type).ask

                # precentage stop loss price calculation for buy
                # sl = slCalPrice - (slCalPrice * self.stopLoss)

                # point stop loss price calculation for buy
                sl = slCalPrice - (self.stop_distance * point)

                # Take profit
                # tp = currentPrice + (currentPrice * self.takeProfit)
                    
            elif(order_type == "sell"):
                order = mt5.ORDER_TYPE_SELL
                currentPrice = mt5.symbol_info_tick(self.currency_type).bid
                
                # precentage stop loss price calculation for buy
                # sl = slCalPrice + (slCalPrice * self.stopLoss)

                # point stop loss price calculation for buy
                sl = slCalPrice + (self.stop_distance * point)

                # Take profit
                # tp = currentPrice - (currentPrice * self.takeProfit)


            # create request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.currency_type,
                "volume": float(self.positionSize),
                "type": order,
                "price": currentPrice,
                "sl": sl,
                "magic": 234000,
                "comment": "Algo trading",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(self.currency_type, "Failed to send order!, retcode={}".format(result.retcode))
                return 0

            else:
                print (self.currency_type, "found!", f"ticket {result.order}, {order_type} Order successfully placed!")
                return result

        except Exception as e:
            print("Exception occured in open_position function:", e)




    # this function send trade close order to trade server
    def close_trade(self, order_type, ticket):

        try:
            # create sutable order type
            if(order_type == "buy"):

                # if running orde_type is buy order is sell
                order = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(self.currency_type).bid

            if(order_type == "sell"):

                # if running orde_type is sell order is buy
                order = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.currency_type).ask


            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.currency_type,
                "volume": self.positionSize,
                "type": order,
                "position": int(ticket),
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Algo trading",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # send a trading request
            result = mt5.order_send(request)


            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(self.currency_type, "Failed to close order!, retcode={}".format(result.retcode))
                return 0

            else:
                print (self.currency_type, order_type, f"ticket {ticket} Order successfully closed!")
                return result


        except Exception as e:
            print("Exception occured in close_trade function:", e)





    # This function will identify trade opens wave
    def identifyWaveNumber(self, date):

        # How many number of peaks current day have when open a trade ?
        # For this use a boolean mask 
        Mask = (self.df['time'] >= date)

        # define new dataframe df3
        df3 = self.df.loc[Mask]
        # print(df3)

        # If there are more than 2 rows in df3
        if len(df3.index) > 2:

            peaksIndexList = []
            for idx, row in df3.loc[df3.first_valid_index()+2: df3.last_valid_index()].iterrows():
                

                # if (currentVolume < previousVolume) and (previousVolume > before previousVolume) previousVolume index is a peak
                if row['tick_volume'] < df3.loc[idx-1].tick_volume and df3.loc[idx-1].tick_volume > df3.loc[idx-2].tick_volume:
                    peaksIndexList.append(idx-1)

            # print(peaksIndexList, len(peaksIndexList))

            # add 1 because,
            # if there are no peaks when trade open we define trade opens from 1 st wave
            # if there are 1 peak already when trade open we define trade opens from 2 wave and so on
            return len(peaksIndexList) + 1

        # If there are less than or equal 2 rows in df3
        elif len(df3.index) <= 2:

            return 1





    # This function help to check, is there are previous open trades in same low or high point (prevent take duplicate trades)
    def execute_trades(self, startTime, orderType, lastslCalPrice):

        # maximam trades algorith can open in trading platform ////////////////////////////////////////////////////////////////////////////

        try:
            # ckeck connection to tarding account
            self.connect(self.accountNumber)     

            # get number of open positions           
            positions_total = mt5.positions_total()

            # close the connection 
            self.disconnect()

        except Exception as e:
            positions_total = 0

         # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        # if positions less than self.maximamTrades
        if positions_total < self.maximumTrades:
            
            # Trade open peak in RBF Volumes, in dataframe this is tick_volume
            # this peak number is important when closing open trades

            # current date
            date = (self.remove_dateTime.date().strftime('%Y-%m-%d') + ' 00:00:00')

            # call identifyWaveNumber function
            tradeOpenwaveNumber = self.identifyWaveNumber(date)
            # print('Trade opens wave number :',tradeOpenwaveNumber)


            # if tradeOpenwaveNumber is 2 or 3
            if tradeOpenwaveNumber == 1 or tradeOpenwaveNumber == 2 or tradeOpenwaveNumber == 3:
                """
                We call openedTrades SQLite3 db, we use db rather than csv file because read, write, update situations.
                """
                conn = sqlite3.connect('Trades.db')

                # create curser object
                c = conn.cursor()


                # check is there any trade open with below conditions
                # after fetch this : if it gives empty list then open trade , else (list not empty <- there is already trade) pass 
                with conn:
                    fetchData = c.execute("SELECT * FROM trades WHERE date_time >=? AND currency = ? AND orderType = ?", (f'{date}', self.currency_type, orderType,)).fetchall()

            
                # return true => fetchData have no data / false => fetchData have data
                if not(fetchData):

                    # Now check is there are running orders from this currency in this self.hGapFactor2 time period
                    with conn:
                        fetchData2 = c.execute("SELECT * FROM trades WHERE date_time >= ? AND currency = ? AND status = ?", (f'{date}', self.currency_type, 'running',)).fetchall()

                    # write trade closing function / close running trades
                    #////////////////////////////////////////////////////////////////////////////////////
                    #if there are data loop through the fetch list and update "running" status to "closed"
                    if fetchData2: 
                        # updata status
                        for fi in fetchData2:

                            # if the running order type is not equal to current orderType then we can close the running orders, ortherwise we have 2 trades with same direction (same order type)
                            if fi[2] != orderType:

                                # ckeck connection to tarding account
                                self.connect(self.accountNumber)

                                try:

                                    # call close_trade function to close open trades
                                    result = self.close_trade(fi[2], fi[5])

                                    # IF SOME REASON FIRST TIME TRADE EXECUTION FAILD TRY 2 ND TIME
                                    if result == 0:
                                        result = self.close_trade(fi[2], fi[5])
                                    
                                except Exception as e:
                                    pass


                                # close the connection 
                                self.disconnect()

                                # update sqLite db
                                with conn:
                                    c.execute("""UPDATE trades SET status = ? WHERE date_time = ? AND currency = ?""", ('closed',fi[0], fi[1],))


                    # write trade execution function
                    #/////////////////////////////////////////////////////////////////////////////////////

                    # ckeck connection to tarding account
                    self.connect(self.accountNumber)

                    # call open_position function
                    result = self.open_position(orderType, lastslCalPrice)

                    # IF SOME REASON FIRST TIME TRADE EXECUTION FAILD TRY 2 ND TIME
                    if result == 0:
                        result = self.open_position(orderType, lastslCalPrice)

                    # Ticket number
                    Ticket = result.order 

                    # close the connection 
                    self.disconnect()

                    # Insert open trade data to sqLite db               
                    with conn:
                      c.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)", (f'{self.remove_dateTime}', self.currency_type, orderType, 'running', tradeOpenwaveNumber, Ticket,))

                    #close the connection
                    conn.close() 

                    return True

                else: 
                    #close the connection
                    conn.close() 

                    return False
                 
            else:
                return False

        else:
            return False




    # This is the tradeds open function
    def trade_open(self):

        # for line in self.lines:
        #     print(line)

        # print("---------------------------------------------------------------------------------------------------------------------\n")

        for line in self.lines:

            # check trade validate range : x current index - x1 or x11 index (for 15 min i choosed 96 / 1 day have 96 15 min candels)
            if (int(self.df.index[-1]) - int(line["x11"])) <= self.hGapFactor1:

                # check derection : if derection is up and the volume is increasing, it's a buy or sell breakout
                if line["direction"] == "up":

                    # check validations <= 2  buy reversels
                    if line["validations"] <= 2:


                        condition = 'N/T'

                        # line dictonary y1 value is the lastLowPoint_for_buy1
                        lastLowPoint_for_buy1 = float(line["y1"])

                        # buy reversel check
                        # Current closing price is grater than last y1 value is a buy check for this :
                        # check closing price pct% change, gapFactor is depend on currency_type volatility
                        # x11 must be in 10/4 = 2 hours 30 minutes range
                        # for buy order trendline length must be grater than trendLen
                        #((self.df.iloc[-1].close - lastLowPoint_for_buy1) / lastLowPoint_for_buy1) * 100
                        if self.VGapFactor2 >= self.precentChange(lastLowPoint_for_buy1, self.df.iloc[-1].close) >= self.VGapFactor1 and self.df.iloc[-1].tick_volume > self.df.loc[line["x1"]].tick_volume and (int(self.df.index[-1]) - int(line["x11"])) <= self.hGapFactor2 and float(line["length"]) >= self.trendLen and lastLowPoint_for_buy1 > self.VWAP().iloc[-1].vwap_tick_volume and lastLowPoint_for_buy1 > self.df['close'].ewm(span=self.MAFactor, adjust=False).mean().iloc[-1]:
                            

                            # in first argumet : self.df.iloc[-1].time is the current time, self.ttimedelta = timedelta(hours=2, minutes=30) is substract 2 hours and 30 minutes from current time
                            # then we get the starting dattime we are looking in openedTrades_df
                            # second argument : is currency_type self.currency_type name we are checking
                            # thired argment : is orderType buy / sell / sell breakout / buy breakout
                            # check is the condition not true ? (there are no trades open in last 2 houres in this currecy, orderType)
                            # Finnaly call execute_trades function that execute the trade
                            return_condition_true_or_false = self.execute_trades((self.df.iloc[-1].time - self.ttimedelta), 'buy', lastLowPoint_for_buy1)

                            if return_condition_true_or_false:
                                condition = "buy.."
            
                                print(line["x1"], condition, line["direction"], line["validations"], self.precentChange(lastLowPoint_for_buy1, self.df.iloc[-1].close),
                                        self.df.iloc[-1].tick_volume, self.df.loc[line["x1"]].tick_volume, (int(self.df.index[-1]) - int(line["x11"])), float(line["length"]))

                    # check validations > 2 sell breakout or buy reversel
                    if line["validations"] > 2:

                        # sort dataframe descending order to catch first intercept closing price
                        df2 = self.df.sort_values(['time'], ascending=[False])

                        # iterate [x11 < .... < current closing price index] this range
                        for idx, row in df2.loc[df2.first_valid_index(): line["x11"]+1].iterrows():

                            # if closing price < mx + b / HERE WE ARE TRYING TO FIGURE OUT INTERCEPT POINT SO THIS IS THiS CONDITION
                            if row["close"] < line["m"] * idx + line["b"]:
                                pass

                            else:
                                condition = 'N/T'

                                # sell break out : uptrend lastLowPoint_for_sell_breakout value will be last y = mx + c intercept closing price
                                # check closing price pct% change, this value must be - pct%
                                # gapFactor is depend on currency_type volatility
                                # row["close"] < self.df['close'].rolling(window=self.MAFactor).mean().iloc[idx] <-- intercept point index closing price must be lower than that index moving average closing price value
                                #((self.df.iloc[-1].close - row['close']) / row['close']) * 100
                                if (self.VGapFactor2 * -1) <= self.precentChange(row['close'], self.df.iloc[-1].close) <= (self.VGapFactor1 * -1) and self.df.iloc[-1].tick_volume > row['tick_volume'] and row["close"] < self.VWAP().iloc[idx].vwap_tick_volume and row["close"] < self.df['close'].ewm(span=self.MAFactor, adjust=False).mean().iloc[idx]:
              
                                     # in first argumet : self.df.iloc[-1].time is the current time, self.ttimedelta = timedelta(hours=2, minutes=30) is substract 2 hours and 30 minutes from current time
                                    # then we get the starting dattime we are looking in openedTrades_df
                                    # second argument : is currency_type self.currency_type name we are checking
                                    # thired argment : is orderType buy / sell / sell breakout / buy breakout
                                    # check is the condition not true ? (there are no trades open in last 2 houres in this currecy, orderType)
                                    # Finnaly call execute_trades function that execute the trade
                                    return_condition_true_or_false = self.execute_trades((self.df.iloc[-1].time - self.ttimedelta), 'sell', row['close'])

                                    if return_condition_true_or_false:
                                        condition = "sell breakout.."

                                        print(idx, condition, line["direction"], line["validations"], df2.first_valid_index(), line["x11"]+1, row["close"], line["m"] * idx + line["b"], self.precentChange(row['close'], self.df.iloc[-1].close), self.df.iloc[-1].tick_volume, row['tick_volume'], self.df.loc[line["x11"]].tick_volume, (int(self.df.index[-1]) - int(line["x11"])), float(line["length"]))
                             



                                # buy reversel check
                                # Current closing price is grater than last y1 value is a buy check for this :
                                # check closing price pct% change, gapFactor is depend on currency_type volatility
                                # x11 must be in 10/4 = 2 hours 30 minutes range
                                # for buy order trendline length must be grater than trendLen

                                # line dictonary y1 value is the lastLowPoint_for_buy2
                                # ((self.df.iloc[-1].close - lastLowPoint_for_buy2) / lastLowPoint_for_buy2) * 100
                                lastLowPoint_for_buy2 = float(line["y1"])
                                if self.VGapFactor2 >= self.precentChange(lastLowPoint_for_buy2, self.df.iloc[-1].close) >= self.VGapFactor1 and self.df.iloc[-1].tick_volume > self.df.loc[line["x11"]].tick_volume and (int(self.df.index[-1]) - int(line["x11"])) <= self.hGapFactor2 and lastLowPoint_for_buy2 > self.VWAP().iloc[idx].vwap_tick_volume and lastLowPoint_for_buy2 > self.df['close'].ewm(span=self.MAFactor, adjust=False).mean().iloc[idx]:
       

                                     # in first argumet : self.df.iloc[-1].time is the current time, self.ttimedelta = timedelta(hours=2, minutes=30) is substract 2 hours and 30 minutes from current time
                                    # then we get the starting dattime we are looking in openedTrades_df
                                    # second argument : is currency_type self.currency_type name we are checking
                                    # thired argment : is orderType buy / sell / sell breakout / buy breakout
                                    # check is the condition not true ? (there are no trades open in last 2 houres in this currecy, orderType)
                                    # Finnaly call execute_trades function that execute the trade
                                    return_condition_true_or_false = self.execute_trades((self.df.iloc[-1].time - self.ttimedelta), 'buy', lastLowPoint_for_buy2)

                                    if return_condition_true_or_false:
                                        condition = "buy reversal.."

                                        print(idx, condition, line["direction"], line["validations"], df2.first_valid_index(), line["x11"]+1, row["close"], line["m"] * idx + line["b"], self.precentChange(lastLowPoint_for_buy2, self.df.iloc[-1].close), self.df.iloc[-1].tick_volume, row['tick_volume'], self.df.loc[line["x11"]].tick_volume, (int(self.df.index[-1]) - int(line["x11"])), float(line["length"]))
                                
                                break

                # check derection : if derection is down and the volume is increasing, it's a sell or buy breakout
                if line["direction"] == "down":

                    # check validations <= 2  buy reversels
                    if line["validations"] <= 2:

                        condition = 'N/T'

                        # line dictonary y1 value is the lastLowPoint_for_sell1
                        lastLowPoint_for_sell1 = float(line["y1"])

                        # sell reversel check
                        # Current closing price is lesser than last y1 value is a sell check for this :
                        # check closing price pct% change, gapFactor is depend on currency_type volatility
                        # x11 must be in 10/4 = 2 hours 30 minutes range
                        # for sell order trendline length must be grater than trendLen
                        #((self.df.iloc[-1].close - lastLowPoint_for_sell1) / lastLowPoint_for_sell1) * 100
                        if (self.VGapFactor2 * -1) <= self.precentChange(lastLowPoint_for_sell1, self.df.iloc[-1].close) <= (self.VGapFactor1 * -1) and self.df.iloc[-1].tick_volume > self.df.loc[line["x1"]].tick_volume and (int(self.df.index[-1]) - int(line["x11"])) <= self.hGapFactor2 and float(line["length"]) >= self.trendLen and lastLowPoint_for_sell1 < self.VWAP().iloc[-1].vwap_tick_volume and lastLowPoint_for_sell1 < self.df['close'].ewm(span=self.MAFactor, adjust=False).mean().iloc[-1]:

                             # in first argumet : self.df.iloc[-1].time is the current time, self.ttimedelta = timedelta(hours=2, minutes=30) is substract 2 hours and 30 minutes from current time
                            # then we get the starting dattime we are looking in openedTrades_df
                            # second argument : is currency_type self.currency_type name we are checking
                            # thired argment : is orderType buy / sell / sell breakout / buy breakout
                            # check is the condition not true ? (there are no trades open in last 2 houres in this currecy, orderType)
                            # Finnaly call execute_trades function that execute the trade
                            return_condition_true_or_false = self.execute_trades((self.df.iloc[-1].time - self.ttimedelta), 'sell', lastLowPoint_for_sell1)

                            if return_condition_true_or_false:
                                condition = "sell.."


                                print(line["x1"], condition, line["direction"], line["validations"], self.precentChange(lastLowPoint_for_sell1, self.df.iloc[-1].close),
                                        self.df.iloc[-1].tick_volume, self.df.loc[line["x1"]].tick_volume, (int(self.df.index[-1]) - int(line["x11"])), float(line["length"]))

                    # check validations > 2 buy breakout or sell reversel
                    if line["validations"] > 2:

                        # sort dataframe descending order to catch first intercept closing price
                        df2 = self.df.sort_values(['time'], ascending=[False])

                        # iterate [x11 < .... < current closing price index] this range
                        for idx, row in df2.loc[df2.first_valid_index(): line["x11"]+1].iterrows():

                            # if closing price < mx + b
                            if row["close"] > line["m"] * idx + line["b"]:
                                pass

                            else:
                                condition = 'N/T'

                                # buy break out : downtrend lastLowPoint_for_buy_breakout value will be last y = mx + c intercept closing price
                                # check closing price pct% change, this value must be + pct%
                                # gapFactor is depend on currency_type volatility
                                #row["close"] > self.df['close'].rolling(window=self.MAFactor).mean().iloc[idx] <-- intercept point index closing price must be grater than that index moving average closing price value
                                # ((self.df.iloc[-1].close - row['close']) / row['close']) * 100
                                if  self.VGapFactor2 >= self.precentChange(row['close'], self.df.iloc[-1].close) >= self.VGapFactor1 and self.df.iloc[-1].tick_volume > row['tick_volume'] and row["close"] > self.VWAP().iloc[idx].vwap_tick_volume and row["close"] > self.df['close'].ewm(span=self.MAFactor, adjust=False).mean().iloc[idx]:


                                     # in first argumet : self.df.iloc[-1].time is the current time, self.ttimedelta = timedelta(hours=2, minutes=30) is substract 2 hours and 30 minutes from current time
                                    # then we get the starting dattime we are looking in openedTrades_df
                                    # second argument : is currency_type self.currency_type name we are checking
                                    # thired argment : is orderType buy / sell / sell breakout / buy breakout
                                    # check is the condition not true ? (there are no trades open in last 2 houres in this currecy, orderType)
                                    # Finnaly call execute_trades function that execute the trade
                                    return_condition_true_or_false = self.execute_trades((self.df.iloc[-1].time - self.ttimedelta), 'buy', row['close'])

                                    if return_condition_true_or_false:
                                        condition = "buy breakout..."

                                        print(idx, condition, line["direction"], line["validations"], df2.first_valid_index(), line["x11"]+1, row["close"], line["m"] * idx + line["b"], self.precentChange(row['close'], self.df.iloc[-1].close), self.df.iloc[-1].tick_volume, row['tick_volume'], self.df.loc[line["x11"]].tick_volume, (int(self.df.index[-1]) - int(line["x11"])), float(line["length"]))
                                

                                # sell reversel check
                                # Current closing price is lesser than last y1 value is a sell check for this :
                                # check closing price pct% change, gapFactor is depend on currency_type volatility
                                # x11 must be in 10/4 = 2 hours 30 minutes range
                                # for sell order trendline length must be grater than trendLen

                                # line dictonary y1 value is the lastLowPoint_for_sell2
                                #((self.df.iloc[-1].close - lastLowPoint_for_sell2) / lastLowPoint_for_sell2) * 100
                                lastLowPoint_for_sell2 = float(line["y1"])
                                if (self.VGapFactor2 * -1) <= self.precentChange(lastLowPoint_for_sell2, self.df.iloc[-1].close) <= (self.VGapFactor1 * -1) and self.df.iloc[-1].tick_volume > self.df.loc[line["x11"]].tick_volume and (int(self.df.index[-1]) - int(line["x11"])) <= self.hGapFactor2 and lastLowPoint_for_sell2 < self.VWAP().iloc[idx].vwap_tick_volume and lastLowPoint_for_sell2 < self.df['close'].ewm(span=self.MAFactor, adjust=False).mean().iloc[idx]:
                                    condition = "sell reversel.."


                                     # in first argumet : self.df.iloc[-1].time is the current time, self.ttimedelta = timedelta(hours=2, minutes=30) is substract 2 hours and 30 minutes from current time
                                    # then we get the starting dattime we are looking in openedTrades_df
                                    # second argument : is currency_type self.currency_type name we are checking
                                    # thired argment : is orderType buy / sell / sell breakout / buy breakout
                                    # check is the condition not true ? (there are no trades open in last 2 houres in this currecy, orderType)
                                    # Finnaly call execute_trades function that execute the trade
                                    return_condition_true_or_false = self.execute_trades((self.df.iloc[-1].time - self.ttimedelta), 'sell', lastLowPoint_for_sell2)

                                    if return_condition_true_or_false:
                                        condition = "sell reversal.."

                                        print(idx, condition, line["direction"], line["validations"], df2.first_valid_index(), line["x11"]+1, row["close"], line["m"] * idx + line["b"], self.precentChange(lastLowPoint_for_sell2, self.df.iloc[-1].close), self.df.iloc[-1].tick_volume, row['tick_volume'], self.df.loc[line["x11"]].tick_volume, (int(self.df.index[-1]) - int(line["x11"])), float(line["length"]))
                                
                                break





class TradeClose:

    def __init__(self, ticketNumber, breakEvenSD: int = 1, TrallingSD: int = 10):

        self.accountNumber: int = "accountNumber" #enter accountNumber as int
        self.ticketNumber = ticketNumber
        self.breakEvenSD = breakEvenSD
        self.TrallingSD = TrallingSD



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



    # initialize conection with trading platform
    def connect(self, account):

        mt5.initialize()
        authorized=mt5.login(int(account))

        if authorized:
            print("Connected: Connecting to MT5 Client")
        else:
            print("Failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))



    # close the connection
    def disconnect(self):  

        # shut down connection to the MetaTrader 5 terminal
        mt5.shutdown()



    # Update database
    def updateDB(self):

        # connect with Trades SQLite3 db
        conn = sqlite3.connect('Trades.db')

        # create curser object
        c = conn.cursor()

        # update running status to closed
        with conn:
            c.execute("""UPDATE trades SET status = ? WHERE ticket = ?""", ('closed',self.ticketNumber,))

        #close the connection
        conn.close()     



    # this function send trade close order to trade server
    def modify_trade(self, positionDf, level):

        order_type = positionDf['type'][0]
        currency_type = positionDf['symbol'][0]

        print(order_type, currency_type, positionDf['sl'][0])

        try:

            symbol_info = mt5.symbol_info(currency_type)
            point = symbol_info.point * 10

            if symbol_info is None:
                print(currency_type, "not found")
                return 

            if not symbol_info.visible:
                # print(self.currency_type, "is not visible, trying to switch on")
                if not mt5.symbol_select(currency_type, True):
                    print("symbol_select({}}) failed, exit",currency_type)
                    return 

            # if running orde_type is buy
            if (0 == int(order_type)):
                
                price = mt5.symbol_info_tick(currency_type).ask

                # check level
                if level == 3:

                    # point stop loss price calculation for level 3 buy
                    currentSL = price - (self.TrallingSD * point)

                    # check is current stop loss (currentSL) is grater than previous stoploss (perviousSL)
                    if positionDf['sl'][0] < currentSL:
                        sl = currentSL
                    else:
                        pass


                elif level == 2:
                    # precentage stop loss price calculation for level 2 buy
                    currentSL = price - (price * 0.001)


                    # check is current stop loss (currentSL) is grater than previous stoploss (perviousSL)
                    if positionDf['sl'][0] < currentSL:
                        sl = currentSL
                    else:
                        pass


                elif level == 1:
                    # point stop loss price calculation for level 1 buy
                    # here sl dinamically changing accouding to volume(position size), this help cut commission
                    # self.breakEvenSD help cut spared
                    currentSL = positionDf['price_open'][0] + (positionDf['volume'][0]* 100  + self.breakEvenSD) * point
                    

                    # check is current stop loss (currentSL) is grater than previous stoploss (perviousSL)
                    if positionDf['sl'][0] < currentSL:
                        sl = currentSL
                    else:
                        pass


                # create request
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": currency_type,
                    "position": int(self.ticketNumber),
                    "sl": sl,
                    "deviation": 20,
                    "comment": "Algo trading / edit SL",
                }

                # send a trading request
                result = mt5.order_send(request)

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(currency_type, "Failed to modify order!, retcode={}".format(result.retcode))
                    return 0

                else:
                    print (currency_type, f"ticket {self.ticketNumber} Order successfully modified!")
                    return result



            # if running orde_type is sell
            elif (1 == int(order_type)):

                price = mt5.symbol_info_tick(currency_type).bid

                # check level
                if level == 3:

                    # point stop loss price calculation for level 3 buy
                    currentSL = price + (self.TrallingSD * point)

                    # check is current stop loss (currentSL) is grater than previous stoploss (perviousSL)
                    if positionDf['sl'][0] > currentSL:
                        sl = currentSL
                    else:
                        pass


                elif level == 2:
                    # precentage stop loss price calculation for level 2 buy
                    currentSL = price + (price * 0.001)

                    # check is current stop loss (currentSL) is grater than previous stoploss (perviousSL)
                    if positionDf['sl'][0] > currentSL:
                        sl = currentSL
                    else:
                        pass


                elif level == 1:
                    # point stop loss price calculation for level 1 buy
                    # here sl dinamically changing accouding to volume(position size), this help cut commission
                    # self.breakEvenSD help cut spared
                    currentSL = positionDf['price_open'][0] - (positionDf['volume'][0]* 100  + self.breakEvenSD) * point
   

                    # check is current stop loss (currentSL) is grater than previous stoploss (perviousSL)
                    if positionDf['sl'][0] > currentSL:
                        sl = currentSL
                    else:
                        pass



                # create request
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": currency_type,
                    "position": int(self.ticketNumber),
                    "sl": sl,
                    "deviation": 20,
                    "comment": "Algo trading / edit SL",
                }

                # send a trading request
                result = mt5.order_send(request)

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(currency_type, "Failed to modify order!, retcode={}".format(result.retcode))
                    return 0

                else:
                    print (currency_type, f"ticket {self.ticketNumber} Order successfully modified!")
                    return result


        except Exception as e:
            print("Exception occured in modify_trade function:", e)




    # This is the trades closing function
    def trade_close(self):
        
        # validate each db trade with MT5 open trades, to check is this trade relly exist or not

        # ckeck connection to tarding account
        self.connect(self.accountNumber)    

        # get open position using ticket number
        position = mt5.positions_get(ticket=self.ticketNumber)


        # if trade does not exist this if will run
        if not position:

            print(f"No position, {self.ticketNumber} trade is already closed")

            # Update the trade
            self.updateDB()
            
        # other wise
        elif len(position) == 1:

            '''
            Then we have 3 status
              1. if current price in 0.01% level -> Move SL to break even (> commission)
              2. if current price in 0.02% level -> Move SL to 0.01% level
              3. if current price in 0.025% level -> Set 10 tralling SL

            '''

            # display these position as a table using pandas.DataFrame
            positionDf = pd.DataFrame(list(position),columns=position[0]._asdict().keys())

            # drop unwanted columns
            positionDf.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id', 'time', 'magic', 'identifier', 'reason', 'tp', 'comment'], axis=1, inplace=True)
            # print(positionDf)

            # print(positionDf)

            # calculate trade current precentage + or - side
            tpPrecenage = self.precentChange(positionDf['price_open'][0], positionDf['price_current'][0])

            print(tpPrecenage)

            # buy or sell
            order_type = positionDf['type'][0]

            # if buy
            if (0 == int(order_type)):

                # 3. Set 10 tralling SL //////////////////////////////////////////////////////////
                if tpPrecenage >= 0.25:

                    # call open_position function
                    result = self.modify_trade(positionDf, 3)

                    # IF SOME REASON FIRST TIME TRADE MODIFICATION FAILD TRY 2 ND TIME
                    if result == 0:
                            result = self.modify_trade(positionDf, 3)

                    # Update the trade
                    self.updateDB()


                # 2. Move SL to 0.01% level ///////////////////////////////////////////////////////
                elif tpPrecenage >= 0.2:

                    # call open_position function
                    result = self.modify_trade(positionDf, 2)

                    # IF SOME REASON FIRST TIME TRADE MODIFICATION FAILD TRY 2 ND TIME
                    if result == 0:
                            result = self.modify_trade(positionDf, 2)


                # 1. Move SL to break even ///////////////////////////////////////////////////////
                elif tpPrecenage >= 0.1:

                    # call open_position function
                    result = self.modify_trade(positionDf, 1)

                    # IF SOME REASON FIRST TIME TRADE MODIFICATION FAILD TRY 2 ND TIME
                    if result == 0:
                            result = self.modify_trade(positionDf, 1)

        

            # if sell
            elif (1 == int(order_type)):

                # 3. Set 10 tralling SL //////////////////////////////////////////////////////////
                if tpPrecenage <= -0.25:

                    # call open_position function
                    result = self.modify_trade(positionDf, 3)

                    # IF SOME REASON FIRST TIME TRADE MODIFICATION FAILD TRY 2 ND TIME
                    if result == 0:
                            result = self.modify_trade(positionDf, 3)

                    # Update the trade
                    self.updateDB()


                # 2. Move SL to 0.01% level ///////////////////////////////////////////////////////
                elif tpPrecenage <= -0.2:

                    # call open_position function
                    result = self.modify_trade(positionDf, 2)

                    # IF SOME REASON FIRST TIME TRADE MODIFICATION FAILD TRY 2 ND TIME
                    if result == 0:
                            result = self.modify_trade(positionDf, 2)


                # 1. Move SL to break even ///////////////////////////////////////////////////////
                elif tpPrecenage <= -0.1:

                    # call open_position function
                    result = self.modify_trade(positionDf, 1)

                    # IF SOME REASON FIRST TIME TRADE MODIFICATION FAILD TRY 2 ND TIME
                    if result == 0:
                            result = self.modify_trade(positionDf, 1)



        else:
            print('Unknown error !!!!!!')      


        # close the connection 
        self.disconnect()






def TradeOpenMain(currency_type):

    #create instance using Currency class
    currencyInstance = Currency(currency_type)

    # lode currency values using get_data function
    condi, df, remove_dateTime, remove_closingPrice = currencyInstance.get_data()   

    if condi:

        startTime = time.time()

        """
        Before the live trade execution we need to go through some validation levels
        First level : check is there are any data, If yes execute the trading model, else pass, THIS IS DONE IN CURRENCY CLASS
        """
            
        # Create instanse from TradingModel Class-------------------------------------------
        modelInstance = TradingModel(df, distance_factor=0.1, n=5, extend_lines=True)

        lines = modelInstance.findTrends()

        # This df different from loadData() function return df, new df only have 2 columns (time, clos, tick_volume=RBF volume)
        df = modelInstance.findRBFVolume()

        # Create instanse from TradeOpen Class----------------------------------------------
        tradeOpenInstance = TradeOpen(df, lines, currency_type, remove_dateTime, positionSize=0.02, maximumTrades=5)

        tradeOpenInstance.trade_open()


        endTime = time.time()
        # print(f"{remove_dateTime} : Process takes :", endTime - startTime, "seconds.")
        # print("-----------------------------------------------------------------------------\n")

        return f"{remove_dateTime} {currency_type}: Process takes : {endTime - startTime} seconds."

        # Call drawing function
        # plotter4.drawChart(df, lines)#, Max_Min ,dfMax, dfMin)



def TradeCloseMain():

    # connect with Trades SQLite3 db
    conn = sqlite3.connect('Trades.db')

    # create curser object
    c = conn.cursor()


    # check is there are running orders
    with conn:
        fetchData = c.execute("SELECT * FROM trades WHERE status = ?", ('running',)).fetchall()

    # return true => fetchData have  data / false => fetchData have no data
    if fetchData:

        # loop throw fetched data
        for data in fetchData:

            #create instance using TradeClose class
            tradeCloseInstance = TradeClose(data[5])

            # lode trade_close function
            tradeCloseInstance.trade_close()

            # time.sleep(1) # add 1 second dilay 

        #close the connection
        conn.close() 

    else:

        #close the connection
        conn.close() 




if __name__ == '__main__':

    previous_time = '0'

    while True:

        #create instance using Currency class
        currencyInstanceMain = Currency('EURUSD', numberof_bars = 1)

        # lode currency values using get_data function
        condMain, dfMain, remove_dateTimeMain, remove_closingPriceMain = currencyInstanceMain.get_data()   


        if condMain:
            
            # extrect minutes as string value
            currentMin = remove_dateTimeMain.time().strftime('%H:%M:%S').split(':')[1]


            # this part runs every 15 minutes //////////////////////////////////////////////////////////////////////////////////////////////
            if remove_dateTimeMain.time().strftime('%H:%M:%S') > '02:25:00' and remove_dateTimeMain.time().strftime('%H:%M:%S') < '15:50:00':

                # if the currentMin value is not equal to previous time minutes then continue with the process
                if currentMin != previous_time:

                    startTime = time.time()

                    # Currency pair list
                    currency_type = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDJPY', 'USDCHF', 'AUDCHF', 'CHFJPY', 'GBPNZD', 
                                     'AUDCAD', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'GBPAUD', 'AUDJPY', 'AUDNZD', 
                                     'CADCHF', 'CADJPY', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'NZDCAD', 'NZDCHF', 'NZDJPY']


                    # mom dictonary
                    mom = {'currency':[], 'v1':[]}

                    for currency in currency_type:

                        #create instance using Currency class
                        currencyInstanceMain2 = Currency(currency, date_from = remove_dateTimeMain.date(), date_to = remove_dateTimeMain.date())

                        # lode currency values using get_data function
                        pctChangeOfCurrency = currencyInstanceMain2.get_price_values()  

                        mom['currency'].append(currency)
                        mom['v1'].append(pctChangeOfCurrency)


                    # generate new currency_type2 list accouding to daily closeing price precentage (generate multiprocess sequence list of currencies)
                    currency_type2 = pd.DataFrame(mom).sort_values(by=['v1'], ascending = False).currency.tolist()
                    # print(currency_type2)


                    # create db Trades if not exists
                    conn = sqlite3.connect('Trades.db')
                    c = conn.cursor()

                    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
                            date_time   TEXT,
                            currency    TEXT,
                            orderType   TEXT,
                            status  TEXT,
                            tradeOpenWave   INTEGER,
                            ticket  INTEGER)""")

                    conn.close()


                    # Multiprossing process
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results = executor.map(TradeOpenMain, currency_type2)

                        for result in results:
                            print(result)


                    endTime = time.time()
                    print(f"{remove_dateTimeMain} Process takes : {endTime - startTime} seconds.\n")

            previous_time = currentMin

            # time.sleep(1)

        TradeCloseMain()
        # break

















        
        
        
