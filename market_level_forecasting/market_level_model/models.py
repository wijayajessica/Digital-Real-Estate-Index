import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import tensorflow.compat.v2 as tf
from tensorflow_probability import sts
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class baseModel(object):
    def __init__(self,df,prediction_horizon,target_column,feature_column_names,lead_target=True):
        self.df = df
        self.prediction_horizon = prediction_horizon
        self.target_column= target_column
        self.feature_column_names = feature_column_names
        if lead_target == True:
            self._calculate_lead_target()
        else:
            self.target_column_names = self.target_column
        self.model = None
        self.selected_features_no = np.ones(len(self.feature_column_names))
        self.selected_features = self.feature_column_names
        self.lead_target = lead_target
        #print('initilized successfully!')

    def _calculate_lead_target(self,):
        self.target_column_names = self.target_column+'_target'
        self.df[self.target_column_names] = -self.df[self.target_column].diff(-self.prediction_horizon)

    def get_train_test(self,test_size,standardise=True):
        merged = self.df.dropna(axis=0)
        #print(len(merged[self.target_column_names]))
        train_X, test_X, train_y, test_y = train_test_split(merged[self.feature_column_names],merged[self.target_column_names], test_size = test_size, random_state=100, shuffle=False)
        #standardize the features
        if standardise == True:
            std = StandardScaler()
            std.fit(train_X)
            train_X_std = std.transform(train_X)
            test_X_std = std.transform(test_X)
            return train_X_std, test_X_std, train_y, test_y
        else:
            return train_X, test_X, train_y, test_y

    def select_features(self,x,y,method='Lasso'): #method: 'Lasso','PCA','Feature Importance'
        if method == 'Lasso':
            lr = LassoCV()
            if self.lead_target == False:
                lr = LassoCV()
            #print(x.shape, y.shape)
            lr.fit(x, y)
            #print(lr.coef_)
            self.selected_features = self.feature_column_names[abs(lr.coef_)>0]
            #print(self.selected_features)
            self.selected_features_no = np.arange(len(self.feature_column_names))[abs(lr.coef_)>0] 
    
    def get_selected_features(self,):
        return self.selected_features, self.selected_features_no

    def standard(self,x):
        self.std = StandardScaler()
        return self.std.fit_transform(x)

    def fit(self,x,y):
        pass

    def predict(self,x):
        pass

    def model_preformance(self,train_X_std, test_X_std, train_y, test_y, target_col,cv=5):
        y_pred_train = self.model.predict(train_X_std[:,self.selected_features_no])
        y_pred_test = self.model.predict(test_X_std[:,self.selected_features_no])
        rst = {'r2_train': self.model.score(train_X_std[:,self.selected_features_no], train_y),
             'r2_test':self.model.score(test_X_std[:,self.selected_features_no], test_y),
             'mse_train': mean_squared_error(train_y,y_pred_train),
             'mse_test': mean_squared_error(test_y,y_pred_test),
            'cv_score':-cross_val_score(self.model, train_X_std[:,self.selected_features_no], train_y, cv=cv,scoring='neg_mean_squared_error').mean()}
        return rst, y_pred_train, y_pred_test

    def set_features_no(self,features_no):
        self.selected_features_no = features_no

class linearModel(baseModel): 
    def fit(self,x,y,select_features= True, standardise=True): 
    
        if standardise == True:
            x = self.standard(x)
        if select_features == True:
            self.select_features(x,y)
        #fit ridge models
        self.model =  RidgeCV()
        if len(self.selected_features) == 0:
            self.model = None
        else:
            self.model.fit(x[:,self.selected_features_no],y)

    def predict(self,x,standardise =True):
        if self.model == None:
            print('Questionable models: not informative features!')
            return np.zeros(x.shape[0])
        else:
            if standardise == True:
                x = self.std.transform(x)
        
            return self.model.predict(x[:,self.selected_features_no])
    
    def get_prediction(self,x,standardise = True):
        diff_predict = self.predict(x,standardise = standardise)
        return x[self.target_column]+diff_predict

class TreeModel(baseModel):
    def fit(self,x,y,select_features= True): 
        if select_features == True:
            self.select_features(x,y)
        #fit tree models
        self.model =  DecisionTreeRegressor(max_depth=8)
        if len(self.selected_features) == 0:
            self.model = None
        else:
            #print(len(self.selected_features_no))
            #print(x.values.shape)
            #print(x.values[:,self.selected_features_no.astype(int)])
            self.model.fit(x.iloc[:,self.selected_features_no.astype(int)],y)

    def predict(self,x):
        if self.model == None:
            return np.zeros(x.shape[0])
        else:      
            return self.model.predict(x.iloc[:,self.selected_features_no.astype(int)])
    
    def get_prediction(self,x):
        diff_predict = self.predict(x)
        return x[self.target_column]+diff_predict   

def build_model(observed_time_series):
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    seasonal = tfp.sts.Seasonal(num_seasons=12, observed_time_series=observed_time_series)
    all_components = [trend,seasonal]

    model = sts.Sum(all_components, observed_time_series=observed_time_series)
    return model

def build_variational_posteriors(model, training_data, nsamples,
                                 optimizer = tf.optimizers.Adam(learning_rate=.05),
                                 plot = True
                                ):
    # Build the variational surrogate posteriors `qs`.
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

    # Allow external control of optimization to reduce test runtimes.
    num_variational_steps = 200 # @param { isTemplate: true}
    num_variational_steps = int(num_variational_steps)
    # Using fit_surrogate_posterior to build and optimize the variational loss function.
    @tf.function(experimental_compile=True)
    def train():
        elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_log_prob(observed_time_series=training_data),
        surrogate_posterior=variational_posteriors,
        optimizer=optimizer,
        num_steps=num_variational_steps)
        return elbo_loss_curve

    elbo_loss_curve = train()
    
    if plot:
        plt.plot(elbo_loss_curve)
        plt.title("ELBO loss curve")
        plt.show()

    # Draw samples from the variational posterior.
    q_samples = variational_posteriors.sample(nsamples)
    
    return q_samples


class BTSM(object):
    def __init__(self,):
        self.model = 0
        self.q_samples = 0
        
    def fit(self,x_train, y_train):
        self.train = x_train
        self.model = build_model(x_train)
        self.q_samples = build_variational_posteriors(self.model, x_train,len(x_train),
                                                optimizer = tf.optimizers.Adam(learning_rate=.1),
                                                plot = False)
           
    def predict(self,x_test,num_samples=50):
        forecast_dist = tfp.sts.forecast(self.model, observed_time_series=self.train,
                                            parameter_samples=self.q_samples, num_steps_forecast=len(x_test))

        forecast_mean, forecast_scale, forecast_samples = (forecast_dist.mean().numpy()[..., 0],
                                                           forecast_dist.stddev().numpy()[..., 0],
                                                           forecast_dist.sample(num_samples).numpy()[..., 0])
        return forecast_mean, forecast_scale, forecast_samples
        

class rollingModel(object):
    def __init__(self,df, predict_horizon,target_column,feature_column_names,modelName='linear',lead_target=True):
        self.models = {}
        if modelName == 'linear':
            for i in range(1,predict_horizon+1): 
                self.models[i] = linearModel(df,i,target_column,feature_column_names,lead_target)
                print(self.models[i])
        elif modelName == 'tree':
            for i in range(1,predict_horizon+1): 
                self.models[i] = TreeModel(df,i,target_column,feature_column_names,lead_target)
        self.df = df
        self.predict_horizon = predict_horizon
        self.target_column= target_column
        self.feature_column_names = feature_column_names
        self.lead_target = lead_target

    def model_split(self,x, y, time_cv, return_idx = False, cv=None): #cv is the k-fold number
        time_periods = len(x)
        if time_cv == None:
            time_cv = int(np.ceil(time_periods//cv))  #time_cv is the time periods length in a single training set
        else:
            cv = int(np.ceil(time_periods//time_cv))
        self.groups_train = {}
        self.groups_test = {}
        self.split_idx = []
        for i in range(cv-1):
            self.groups_train[i] = (x[:(i+1)*time_cv], y[:(i+1)*time_cv])
            self.groups_test[i] = (x[(i+1)*time_cv:(i+2)*time_cv], y[(i+1)*time_cv:(i+2)*time_cv])
            self.split_idx.append((i+1)*time_cv)
        self.groups_train[cv-1] = (x[:cv*time_cv], y[:cv*time_cv])
        self.groups_test[cv-1] = (x[cv*time_cv:], y[cv*time_cv:])
        self.split_idx.append(cv*time_cv) #split_idx is the index of the begining of test groups
        if return_idx == False:
            return self.groups_train, self.groups_test
        else:
            return self.groups_train, self.groups_test, self.split_idx

    def fit(self,x_train, y_train, start_fit=3):
        self.start_fit = start_fit
        #print('horizon', self.predict_horizon)
        for i in range(1,self.predict_horizon+1):
            #print('prediction_horizon', i)
            self.x_train = x_train
            self.y_train = y_train
            if self.lead_target == True:
                y = y_train.diff(i).shift(-i)[start_fit:-i]
            else:
                y = y_train.shift(-i)[start_fit:-i]
            x = x_train[start_fit:-i]
            #print(self.models[i])
            self.models[i].fit(x,y)

    def predict(self,):
        y_pred = []
        for i in range(1,self.predict_horizon+1):
            pred_diff = self.models[i].predict(self.x_train[-self.start_fit:])
            if self.lead_target == True:
                y_pred.append(pred_diff[-1]+self.y_train.iloc[-1])
            else:
                y_pred.append(pred_diff[-1])
        return y_pred
    
    def fit_predict(self, x_train, y_train, start_fit=3, num_samples=50):
        self.start_fit = start_fit
        #print('horizon', self.predict_horizon)
        total_pred_scale = []
        total_pred_samples = []
        total_pred_mean = []
        preds = []
        for i in range(1,self.predict_horizon+1):
            print('prediction_horizon', i)
            self.x_train = x_train
            self.y_train = y_train
            if self.lead_target == True:
                y = y_train.diff(i).shift(-i)[start_fit:-i]
            else:
                y = y_train.shift(-i)[start_fit:-i]
            x = x_train[start_fit:-i]
            self.models[i].fit(x,y)
            pred_diff = self.models[i].predict(self.x_train[-self.start_fit:])
            if self.lead_target == True:
                preds.append(pred_diff[-1]+self.y_train.iloc[-1])
            else:
                preds.append(pred_diff[-1])
            #print(self.models[i])
            pred_samples = []
            for sam in range(num_samples):
                sam_idx = np.random.choice(np.arange(len(y)),size=len(y))
                x_sam = x.iloc[sam_idx,:]
                y_sam = y.iloc[sam_idx]
                #print(x_sam, y_sam)
                self.models[i].fit(x_sam,y_sam)
                pred_diff = self.models[i].predict(self.x_train[-self.start_fit:])
                if self.lead_target == True:
                    pred_samples.append(pred_diff[-1]+self.y_train.iloc[-1])
                else:
                    pred_samples.append(pred_diff[-1])
            total_pred_scale.append(np.array(pred_samples).std())        
            total_pred_mean.append(np.array(pred_samples).mean())
            total_pred_samples.append(pred_samples)
        return preds, np.array(total_pred_mean), np.array(total_pred_scale), np.array(total_pred_samples).T  

    def rolling_prediction(self,start_predict_group,end_predict_group,time_cv,bootstrap=False):
        groups_train, groups_test= self.model_split(self.df[self.feature_column_names],self.df[self.target_column],time_cv=self.predict_horizon,return_idx=False)
        pred_ahead_store = []
        pred_std_store = []
        #pred_ahead_score = []
        for i in range(start_predict_group,end_predict_group):
            #print('start group', i)
            x = groups_train[i][0]
            y = groups_train[i][1]
            pred_ahead, pred_mean, pred_std, pred_samples = self.fit_predict(x,y)
            #true_predict_length = len(groups_test[i][1])
            #score = mean_squared_error(groups_test[i][1],pred_ahead[:true_predict_length])
            pred_ahead_store.extend(pred_ahead)
            pred_std_store.extend(pred_std)
            #pred_ahead_score.append(score)
        return pred_ahead_store, pred_std_store#pred_ahead_score


class rollingTimeSeriesModel(BTSM,rollingModel):
    def __init__(self,df, predict_horizon,target_column):
        self.df = df
        self.predict_horizon = predict_horizon
        self.target_column = target_column
    
    def rolling_prediction(self,start_predict_group,end_predict_group):
        groups_train, groups_test= rollingModel.model_split(self,self.df[self.target_column],self.df[self.target_column],time_cv=self.predict_horizon,return_idx=False)
        pred_ahead_store = []
        pred_ahead_score = []
        pred_ahead_scale = []
        for i in range(start_predict_group,end_predict_group):
            x = groups_train[i][0].values.astype(float)
            y = groups_train[i][1].values.astype(float)
            #print('x',x)
            #print('y',y)
            BTSM.fit(self,x,y)
            pred_ahead, pred_scale, pred_samples = BTSM.predict(self,np.ones(self.predict_horizon))
            true_predict_length = len(groups_test[i][1])
            score = mean_squared_error(groups_test[i][1],pred_ahead[:true_predict_length])
            pred_ahead_store.extend(pred_ahead)
            pred_ahead_scale.extend(pred_scale)
            pred_ahead_score.append(score)
        return pred_ahead_store, pred_ahead_score, pred_ahead_scale


class rollingCombinedModel(BTSM, rollingModel):
    def __init__(self,df,predict_horizon_total, predict_horizon, feature_column_names, target_column,modelName='linear', lead_target=True, params=[0.5,0.5]):
        rollingModel.__init__(self,df,predict_horizon, target_column, feature_column_names, modelName,lead_target)
        BTSM.__init__(self,)
        self.df = df
        self.predict_horizon_total = predict_horizon_total
        self.predict_horizon = predict_horizon
        self.target_column = target_column
        self.feature_column_names = feature_column_names
        self.params = params
    def rolling_prediction(self,start_predict_group, end_predict_group):
        groups_train, groups_test = rollingModel.model_split(self,self.df[self.feature_column_names],self.df[self.target_column],time_cv=self.predict_horizon_total,return_idx=False)
        pred_ahead_store = []
        pred_samples_store = []
        pred_scale_store = []
        for i in range(start_predict_group,end_predict_group):
            x = groups_train[i][0]
            y = groups_train[i][1]
            #print(x)
            rollingModel.fit(self,x,y)
            #print('fit good!')
            pred, pred_1, pred1_scale, pred1_samples = rollingModel.fit_predict(self,x,y)
            BTSM.fit(self,y.values.astype(float),0)
            pred_ts, pred_scale_ts, pred_samples_ts = BTSM.predict(self,np.ones(self.predict_horizon))
            pred_com = self.params[0]*np.array(pred_1)+self.params[1]*pred_ts
            pred_com_scale = self.params[0]*pred1_scale+self.params[1]*pred_scale_ts
            #print('predict good!')
            x = np.append(groups_train[i][1].values.astype(float), pred_com)
            BTSM.fit(self,x,0)
            pred_2, pred2_scale, pred2_samples = BTSM.predict(self,np.ones(self.predict_horizon_total-self.predict_horizon))
            pred_ahead_store.extend(np.append(pred_com, pred_2))
            #print('ts shape.', np.array(pred2_samples).shape)
            #print('1 shape', np.array(pred1_samples).shape)
            pred_samples_store.extend(np.hstack((pred1_samples, pred2_samples)))
            pred_scale_store.extend(np.append(pred_com_scale, pred2_scale))
            print(i)
        return pred_ahead_store, pred_scale_store, pred_samples_store
    def make_prediction(self,x,y):
        rollingModel.fit(self,x,y)
        pred, pred_1, pred1_scale, pred1_samples = rollingModel.fit_predict(self,x,y)
        BTSM.fit(self,y,0)
        pred_ts, pred_scale_ts, pred_samples_ts = BTSM.predict(self,np.ones(self.predict_horizon))
        pred_com = self.params[0]*np.array(pred_1)+self.params[1]*pred_ts
        pred_com_scale = self.params[0]*pred1_scale+self.params[1]*pred_scale_ts
        #pred_com_samples = 
        x = np.append(y.values.astype(float), pred_com)
        BTSM.fit(self,x,0)
        pred_2, pred2_scale, pred2_samples = BTSM.predict(self,np.ones(self.predict_horizon_total-self.predict_horizon))
        return np.append(pred_com, pred_2), np.append(pred_com_scale, pred2_scale), np.hstack((pred1_samples, pred2_samples))

  
