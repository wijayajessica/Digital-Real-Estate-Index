from io import BytesIO, StringIO
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_poisson_deviance
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance, DMatrix
from lightgbm import LGBMRegressor, plot_importance
from sklearn.tree import DecisionTreeClassifier

# parameter tuning
from itertools import product
from collections import OrderedDict, defaultdict
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from utils import *

# ------------------------------------------------------------------------------------


# for XGBoost parameter tuning
XGB_space= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.01)),
    'max_depth': hp.quniform("max_depth", 5, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 500, 2000, 100),
    'min_child_weight': 1.12, #not much impact
    'colsample_bytree': 0.62, #not much impact
    'colsample_bylevel': 0.7, #not much impact
    'objective':'reg:squarederror',
    'reg_lambda': hp.uniform('reg_lambda', 0.1, 1.0),
    'reg_alpha' : hp.quniform('reg_alpha', 5, 20, 1),
}


class XGB_model:
	def __init__(self, df, target, hyperparameter=None, model=None, space=XGB_space, max_evals=50):

		self.space = space
		self.hyperparameter = hyperparameter
		self.model = model
		self.max_evals = max_evals
		self.df = df
		self.target = target

		preprocessed_data = preprocess_train_test_data(self.df, target_column=self.target)
		self.X_train_normalized, self.X_test_normalized, self.y_train, self.y_test = preprocessed_data #tuple unpacking
		self.df_test_pred = self.df.loc[self.X_test_normalized.index]

	def XGB_hyperparameter_tuning(self, space):
	    model = XGBRegressor(learning_rate = space['learning_rate'],
	                         n_estimators = int(space['n_estimators']), 
	                         max_depth = int(space['max_depth']), 
	                         min_child_weight = space['min_child_weight'],
	                         colsample_bytree = space['colsample_bytree'],
	                         colsample_bylevel = space['colsample_bylevel'],
	                         objective = space['objective'],
	                         reg_lambda = space['reg_lambda'],
	                         reg_alpha = int(space['reg_alpha']), 
	                         )
	    
	    evaluation = [(self.X_train_normalized, self.y_train), (self.X_test_normalized, self.y_test)]
	    
	    model.fit(self.X_train_normalized, self.y_train, eval_set=evaluation, eval_metric="mae", early_stopping_rounds=3, verbose=False)

	    pred = model.predict(self.X_test_normalized)
	    error= mean_absolute_error(self.y_test, pred)

	    return {'loss':error, 'status': STATUS_OK, 'model': model}

	def find_bestparams(self):
		self.hyperparameter = fmin(fn = self.XGB_hyperparameter_tuning, space=self.space, algo=tpe.suggest, 
			max_evals=self.max_evals, trials= Trials())

	def build_model(self):
		if self.hyperparameter == None:
			return "Pass in the hyperparameter to use, or tune hyperparameter from scratch by executing self.find_bestparams()"

		self.model = XGBRegressor(learning_rate = self.hyperparameter['learning_rate'], 
			max_depth = int(self.hyperparameter['max_depth']), 
			n_estimators = int(self.hyperparameter['n_estimators']),
			reg_alpha = int(self.hyperparameter['reg_alpha']), reg_lambda = self.hyperparameter['reg_lambda'], 
			min_child_weight = self.space['min_child_weight'], 
			colsample_bytree = self.space['colsample_bytree'], colsample_bylevel = self.space['colsample_bylevel'], 
			objective = self.space['objective'], eval_metric = 'mae', base_score = np.median(self.y_train))

		print(self.model)

		self.model.fit(self.X_train_normalized, self.y_train)
		print("\nMAE training:", mean_absolute_error(self.y_train, self.model.predict(self.X_train_normalized)))
		print("MAE testing:", mean_absolute_error(self.y_test, self.model.predict(self.X_test_normalized)))

	def evaluate_model(self, show_plot=True, ax=None):

		if self.model == None:
			return "No model provided. Set self.model=model or train a model from scratch by executing self.build_model()"

		
		self.df_test_pred['pred'] = self.model.predict(self.X_test_normalized)
		self.df_test_pred['pred'] = self.model.predict(self.X_test_normalized)

		XGB_mean_residuals = []

		for ct in self.df_test_pred.ct_key.unique():
		  mean = np.mean(self.df[self.df['ct_key']==ct][self.target])
		  pred_XGB = np.mean(self.df_test_pred[self.df_test_pred['ct_key']==ct]['pred'])
		  XGB_mean_residuals.append(pred_XGB-mean)
		  
		print(f"XGboost - mean:{np.mean(XGB_mean_residuals):.5f}, st dev:{np.std(XGB_mean_residuals):.5f}")

		if not ax:
			fig, ax=plt.subplots(1,1,figsize=(7,5))

		ax.hist(XGB_mean_residuals, bins=20, alpha=0.5, label='distributions of residuals')
		ax.axvline(np.mean(XGB_mean_residuals), c='k', label='mean')
		ax.legend()
		ax.set_title("difference in XGB prediction vs mean for eact CT")

		if show_plot:
			plt.show()

		return XGB_mean_residuals

	def plot_pred_CT(self, samples_ct, ax, plot_distribution=True):
		if "pred" in self.df_test_pred.columns:
			for idx, ct in enumerate(samples_ct):
				
				if plot_distribution:
					# plot the distribution of known rates
					ax[idx//5, idx%5].hist(self.df[self.df['ct_key']==ct][self.target], alpha=0.5, label='actual rates')
					ax[idx//5, idx%5].axvline(np.mean(self.df[self.df['ct_key']==ct][self.target]), c='k', linestyle=':', label='mean ratio')

				# plot the prediction
				pred = self.df_test_pred[self.df_test_pred['ct_key']==ct]['pred'].values
				ax[idx//5, idx%5].axvline(np.mean(pred), c='b',label='XGB pred') #taking the mean because we can have > 1 test points belonging to the same census tract
				ax[idx//5, idx%5].legend()

			plt.tight_layout()

		else:
			return "please run self.evaluate_model() first"

	def plot_feature_importance(self, ax):
		if self.model == None:
			return "No model provided. Set self.model=model or train a model from scratch by executing self.build_model()"

		else:
			# XGboost
			sorted_idx = self.model.feature_importances_.argsort()[::-1][:20]
			top20features_XGB = self.X_train_normalized.columns[sorted_idx][::-1]
			ax.barh(top20features_XGB, self.model.feature_importances_[sorted_idx][::-1])
			ax.set_title("XGBoost")


	def plot_feature_contribution(self, sample_idx, n_features = 30, ylabel=""):
		# output prediction (contribution of each feature for each test data)
		booster = self.model.get_booster()
		test_X = DMatrix(self.X_test_normalized, feature_names=self.X_test_normalized.columns)
		listing_predictions_contrib_XGB = booster.predict(test_X, pred_contribs=True)

		# plot contribution
		fig, axes = plt.subplots(2,1, figsize=(15,10), sharex=True, sharey=True)
		fig.suptitle(f"Contribution of the first {n_features} features in XGB model", fontsize=16)
		top_features = self.model.feature_importances_.argsort()[::-1][:n_features] #select top 30 features chosen by XGB model
		
		axes[0].set_title(f"Census Tract {self.df_test_pred.iloc[sample_idx[0]]['ct_key']}")
		axes[0].set_ylabel(ylabel)
		plot_contribution(sample_idx[0], axes[0], top_features, listing_predictions_contrib_XGB, self.X_test_normalized)
		axes[1].set_title(f"Census Tract {self.df_test_pred.iloc[sample_idx[1]]['ct_key']}")
		axes[1].set_ylabel(ylabel)
		plot_contribution(sample_idx[1], axes[1], top_features, listing_predictions_contrib_XGB, self.X_test_normalized)
            


# ------------------------------------------------------------------------------------

# for LightGBM parameter tuning
LGBM_space= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.01)),
    'num_leaves': hp.quniform('num_leaves', 50, 500, 2), #replace max_depth in XGB with num_leaves in LGBM
    'n_estimators': hp.quniform('n_estimators', 500, 2000, 100),
    'min_child_weight': 1.12, #not much impact
    'colsample_bytree': 0.62, #not much impact
    'objective':'regression', 
    'reg_lambda': hp.uniform('reg_lambda', 0.1, 1.0),
    'reg_alpha' : hp.quniform('reg_alpha', 5, 20, 1),
}

class LGBM_model:
	def __init__(self, df, target, hyperparameter=None, model=None, space=LGBM_space, max_evals=50):

		self.space = space
		self.hyperparameter = hyperparameter
		self.model = model
		self.max_evals = max_evals
		self.df = df
		self.target = target

		preprocessed_data = preprocess_train_test_data(self.df, target_column=self.target)
		self.X_train_normalized, self.X_test_normalized, self.y_train, self.y_test = preprocessed_data #tuple unpacking
		self.df_test_pred = self.df.loc[self.X_test_normalized.index]

	def LGBM_hyperparameter_tuning(self, space):
		model =LGBMRegressor(learning_rate = space['learning_rate'],
							n_estimators = int(space['n_estimators']), 
							num_leaves = int(space['num_leaves']), 
							min_child_weight = space['min_child_weight'],
							colsample_bytree = space['colsample_bytree'],
							objective = space['objective'],
							reg_lambda = space['reg_lambda'],
							reg_alpha = int(space['reg_alpha']))

		evaluation = [(self.X_train_normalized, self.y_train), (self.X_test_normalized, self.y_test)]

		model.fit(self.X_train_normalized, self.y_train, eval_set=evaluation, eval_metric="mae", early_stopping_rounds=3, verbose=False)

		pred = model.predict(self.X_test_normalized)
		error= mean_absolute_error(self.y_test, pred)

		return {'loss':error, 'status': STATUS_OK, 'model': model}

	def find_bestparams(self):
		self.hyperparameter = fmin(fn = self.LGBM_hyperparameter_tuning, space=self.space, algo=tpe.suggest, 
			max_evals=self.max_evals, trials= Trials())

	def build_model(self):
		if self.hyperparameter == None:
			return "Pass in the hyperparameter to use, or tune hyperparameter from scratch by executing self.find_bestparams()"
		
		self.model = LGBMRegressor(learning_rate = self.hyperparameter['learning_rate'], 
			num_leaves = int(self.hyperparameter['num_leaves']), n_estimators = int(self.hyperparameter['n_estimators']),
			reg_alpha = int(self.hyperparameter['reg_alpha']), reg_lambda = self.hyperparameter['reg_lambda'], 
			min_child_weight = self.space['min_child_weight'], 
			colsample_bytree = self.space['colsample_bytree'], 
			objective = self.space['objective'])


		print(self.model)

		self.model.fit(self.X_train_normalized, self.y_train, eval_metric='mae')
		print("\nMAE training:", mean_absolute_error(self.y_train, self.model.predict(self.X_train_normalized)))
		print("MAE testing:", mean_absolute_error(self.y_test, self.model.predict(self.X_test_normalized)))

	def evaluate_model(self, show_plot=True, ax=None):

		if self.model == None:
			return "No model provided. Set self.model=model or train a model from scratch by executing self.build_model()"
		
		self.df_test_pred['pred'] = self.model.predict(self.X_test_normalized)
		self.df_test_pred['pred'] = self.model.predict(self.X_test_normalized)

		LGBM_mean_residuals = []

		for ct in self.df_test_pred.ct_key.unique():
		  mean = np.mean(self.df[self.df['ct_key']==ct][self.target])
		  pred_LGBM = np.mean(self.df_test_pred[self.df_test_pred['ct_key']==ct]['pred'])
		  LGBM_mean_residuals.append(pred_LGBM-mean)
		  
		print(f"LGBM - mean:{np.mean(LGBM_mean_residuals):.5f}, st dev:{np.std(LGBM_mean_residuals):.5f}")

		if not ax:
			fig, ax=plt.subplots(1,1,figsize=(7,5))

		ax.hist(LGBM_mean_residuals, bins=20, alpha=0.5, color='r', label='distributions of residuals')
		ax.axvline(np.mean(LGBM_mean_residuals), c='k', label='mean')
		ax.legend()
		ax.set_title("difference in LGBM prediction vs mean for eact CT")

		if show_plot:
			plt.show()

		return LGBM_mean_residuals

	def plot_pred_CT(self, samples_ct, ax, plot_distribution=True):
		if "pred" in self.df_test_pred.columns:
			for idx, ct in enumerate(samples_ct):

				if plot_distribution:
					# plot the distribution of known rates
					ax[idx//5, idx%5].hist(self.df[self.df['ct_key']==ct][self.target], alpha=0.5, label='actual rates')
					ax[idx//5, idx%5].axvline(np.mean(self.df[self.df['ct_key']==ct][self.target]), c='k', linestyle=':', label='mean ratio')

				# plot the prediction
				pred = self.df_test_pred[self.df_test_pred['ct_key']==ct]['pred'].values
				ax[idx//5, idx%5].axvline(np.mean(pred), c='r',label='LGBM pred') #taking the mean because we can have > 1 test points belonging to the same census tract
				ax[idx//5, idx%5].legend()

			plt.tight_layout()

		else:
			return "please run self.evaluate_model() first"

	def plot_feature_importance(self, ax):
		if self.model == None:
			return "No model provided. Set self.model=model or train a model from scratch by executing self.build_model()"

		else:
			sorted_idx = self.model.feature_importances_.argsort()[::-1][:20]
			top20features_LGBM = self.X_train_normalized.columns[sorted_idx][::-1]
			ax.barh(top20features_LGBM, self.model.feature_importances_[sorted_idx][::-1])
			ax.set_title("Light GBM")


	def plot_feature_contribution(self, sample_idx, n_features = 30, ylabel=""):

		# output prediction (contribution of each feature for each test data)
		listing_predictions_contrib_LGBM = self.model.predict(self.X_test_normalized, pred_contrib=True)

		# plot contribution
		fig, axes = plt.subplots(2,1, figsize=(15,10), sharex=True, sharey=True)
		fig.suptitle(f"Contribution of the first {n_features} features in LGBM model", fontsize=16)
		top_features = self.model.feature_importances_.argsort()[::-1][:n_features] #select top 30 features chosen by LGBM model
		
		axes[0].set_title(f"Census Tract {self.df_test_pred.iloc[sample_idx[0]]['ct_key']}")
		axes[0].set_ylabel(ylabel)       
		plot_contribution(sample_idx[0], axes[0], top_features, listing_predictions_contrib_LGBM, self.X_test_normalized)
		axes[1].set_title(f"Census Tract {self.df_test_pred.iloc[sample_idx[1]]['ct_key']}")
		axes[1].set_ylabel(ylabel)      
		plot_contribution(sample_idx[1], axes[1], top_features, listing_predictions_contrib_LGBM, self.X_test_normalized)




