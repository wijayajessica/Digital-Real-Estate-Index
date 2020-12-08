import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler

class getGoogleTrendsData(object):
    def __init__(self,geo='US-CO',city='Atlanta'):
        self.geo = geo
        self.city = city
        self.keywords = [
    ['homes for sale', 'homes for sale in', 'properties for sale','for sale home','homes for sales'], #home for sale
    ['townhomes for sale','townhouses for sale','townhouse for sale'], #townhomes for sale
    'Homes in', 
    'townhouses in',  
    'Homes near me', 
    'condominium for sale near me', 

    'zillow homes for sale',
                
    ['selling a house','sell home', 'selling your home'], #sell homes
     
    ['house appraisal','home appraisal'],#home appraisal
    
    ['home values', 'home valuations', 'home value', 'value of home'],#home values
    ['house valuations','house value', 'house valuation', 'value of house'],#house values
    'property value',
    'real estate values', 
    
    ['home value estimator','estimate property value'],  #home value estimator
     
    ['how much is my house','how much is my house worth',  'price my house', 'how much can i sell my house for'],  #how much is my house
    'home worth',
       
    ['house prices','house price', 'home prices', 'sold home prices'],#home price
                 
    ['rexhomes', 'rex real estate', 'rex home', 'rex house','rex homes'], #rex home
    ['zillow', 'zillow com','zillows'], #zillow.com
    ['real estate agents',  'listing agent'], #agents
    'best realtors',
    ['low commission real estate agents','low commission real estate'],   #low commission real estate
    'how to sell a house without a realtor',
    
                 
    ['for sale by owner', 'fsbo'], #for sale by owner
    'redfin', 
    'trulia', 
    'realtor.com',
    
    ['Homes in '+str(city), 'townhouses in '+str(city),'houses in '+str(city)],#homes in local region
    ['selling a house in '+str(city),'sell home in '+str(city), 'selling your home in '+str(city),'homes for sale in '+str(city)],#Homes for sale in denver             
    ['house appraisal in '+str(city), 'home values in '+str(city), 'house price in '+str(city)],  #house appraisal in denver                
    'for sale by owner in '+str(city)]

    def set_keywords(self, keywords):
        self.keywords = keywords
    
    def get_keyword_to_time_series(self,return_columns = False):
        keyword_to_time_series = {}
        grouper = pd.Grouper(level='date', freq='M')
        keywords_columns = []
        for keyword in self.keywords:
            if type(keyword) == list:
                pytrends = TrendReq(hl='en-US', tz=300)
                pytrends.build_payload(keyword, cat=0, timeframe=self.date, geo=self.geo)
                pytrends_df = pytrends.interest_over_time()
                if len(pytrends_df)!=0: 
                    pytrends_monthly_df = pytrends_df.groupby(grouper)[keyword].mean().reset_index()
                    time_series = np.mean(np.array(pytrends_monthly_df[keyword]),axis=1)
                    keyword_to_time_series[keyword[0]] = time_series
                    keywords_columns.append(keyword[0])
                    #print(keyword)
            else:
                pytrends = TrendReq(hl='en-US', tz=300)
                pytrends.build_payload([keyword], cat=0, timeframe=self.date, geo=self.geo)
                pytrends_df = pytrends.interest_over_time()
                if len(pytrends_df)!=0: 
                    pytrends_monthly_df = pytrends_df.groupby(grouper)[keyword].mean().reset_index()
                    time_series = np.array(pytrends_monthly_df[keyword])
                    keyword_to_time_series[keyword] = time_series
                    keywords_columns.append(keyword)

        if return_columns == True:
            return keyword_to_time_series, keywords_columns
        else:
            return keyword_to_time_series

    def get_data(self,times=1,date='2016-03-01 2020-09-30',scaled=True):
        self.date = date
        keyword_to_time_series, keywords_columns = self.get_keyword_to_time_series(return_columns=True)
        if times >= 1:
            for i in range(times-1):
                keyword_to_time_series_1 = self.get_keyword_to_time_series()
                for keyword in keywords_columns:
                    keyword_to_time_series[keyword] = keyword_to_time_series[keyword]+keyword_to_time_series_1[keyword]    
        df = pd.DataFrame(keyword_to_time_series)
        if scaled == True:
            scaler = MinMaxScaler(feature_range=(0,100))
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            return df_scaled
        else:
            return df

