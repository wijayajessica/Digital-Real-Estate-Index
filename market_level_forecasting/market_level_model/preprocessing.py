import numpy as np
import pandas as pd

class preprocessingMLS(object):
    def __init__(self,dataPath):
        self.df = pd.read_csv(dataPath)
        try:
            a = self.df['list_date']
            b = self.df['sale_date']
        except ValueError:
            print('No columns named list_date/sale_date!')
        self._process_mls()

    def _process_mls(self):
        listData = self.df
        listData['sale_month'] = listData.sale_date.apply(lambda x: str(x)[:-3])
        listData['list_month'] = listData.list_date.apply(lambda x: str(x)[:-3])
        listMonth = listData.groupby(['list_month']).agg({"property_id":"count"}).reset_index()
        saleMonth = listData.groupby(['sale_month']).agg({"property_id":"count","sale_price":"mean"}).reset_index() 
        self.monthData = pd.merge(saleMonth,listMonth, left_on='sale_month',right_on='list_month')
        self.monthData = self.monthData.rename(columns={"sale_month":"month","property_id_x":"count_sale", "property_id_y":"count_list"})
        self.monthData = self.monthData.drop(['list_month'],axis=1)
    
    def get_monthly_data(self,columns=['count_sale','count_list','sale_price'],start_date='2016-03', end_date='2020-09'):
        monthData = self.monthData
        monthData.index = monthData.month
        processed_mls = (monthData.loc[start_date:end_date,columns]).reset_index()
        return processed_mls



class featureEngineering(object):
    def __init__(self,df,raw_feature_columns):
        self.df = df
        self.raw_feature_columns = raw_feature_columns
        self.feature_column_names = self.raw_feature_columns

    def create_lag_features(self,lag_num_list=[1,3]):
        f = self.feature_column_names
        for col in f:
            for n in lag_num_list:
                self.df[col+'_lag'+str(n)] = self.df[col].diff(n)
                self.feature_column_names = np.append(self.feature_column_names,str(col)+'_lag'+str(n))

    def create_pct_change_features(self,lag_num_list=[1,]):
        f = self.raw_feature_columns
        for col in f:
            for n in lag_num_list:
                self.df[col+'_pct'+str(n)] = (self.df[col]+100).pct_change(periods = n)
                self.feature_column_names = np.append(self.feature_column_names,str(col)+'_pct'+str(n))

    def create_month_one_hot(self,):
        self.df['t'] = self.df['month'].apply(lambda x: str(x[-2:])+'m')
        one_hot_month = pd.get_dummies(self.df['t'])
        self.feature_column_names = np.append(self.feature_column_names,one_hot_month.columns)
        self.df = self.df.join(one_hot_month)

    def get_dataFrame(self,):
        return self.df 
        
    def get_feature_names(self,):
        return self.feature_column_names

        
