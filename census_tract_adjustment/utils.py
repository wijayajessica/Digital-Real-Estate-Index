from io import BytesIO, StringIO
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(url):
	'''
	Takes in url of the shared csv file in google drive, and return a dataframe
	Note: sharing setting of the file in google drive has to be "public" 
	'''

	file_id = url.split('/')[-2]
	dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
	df = pd.read_csv(dwn_url)
	
	return df

def preprocess_data(df, listings):

	# preprocess listing dataset
	listings['ct_key'] = listings['ct_key'].astype('int')
	listings['sale_month'] = pd.to_datetime(listings['sale_date'])+ pd.offsets.MonthBegin(n=1) - pd.offsets.MonthBegin(n=1)
	listings['list_month'] = pd.to_datetime(listings['list_date'])+ pd.offsets.MonthBegin(n=1) - pd.offsets.MonthBegin(n=1)

	# get the numbers for listings and sales by census tract
	num_listings_ct = listings.dropna(subset=['list_month']).groupby(['ct_key', 'list_month']).count()[['property_id']].reset_index()
	num_listings_ct = num_listings_ct.rename(columns={'property_id':'ct_count_listings', 'list_month':'month'})
	num_sales_ct = listings.dropna(subset=['sale_month']).groupby(['ct_key', 'sale_month']).count()[['property_id']].reset_index()
	num_sales_ct = num_sales_ct.rename(columns={'property_id':'ct_count_sales', 'sale_month':'month'})
	num_sales_listings = pd.merge(num_listings_ct, num_sales_ct, left_on=['month','ct_key'], right_on=['month','ct_key'], how='outer')
	num_sales_listings = num_sales_listings.fillna(0.)

	# merge with census-tract features dataframe
	df_merged = pd.merge(df, num_sales_listings, left_on='ct_key', right_on='ct_key')

	# get the listings rate & sales rate by month for each census tract, as well as the overall denver's listing & sales rate 
	listing_sales_overall = df_merged.groupby(['ct_key','month']).agg({'ct_count_listings':'mean', 'ct_count_sales':'mean'}).reset_index()
	listing_sales_overall = listing_sales_overall.groupby('month').agg({'ct_count_listings':'sum', 'ct_count_sales':'sum'})

	# sum up number of households for all CT in the house listing dataset and the CT-level feature dataset
	listing_sales_overall['total_households'] = df[df['ct_key'].isin(num_sales_listings.ct_key.unique())]['total_households'].sum()
	listing_sales_overall['sales_per_households'] = listing_sales_overall['ct_count_sales']/listing_sales_overall['total_households']
	listing_sales_overall['listings_per_households'] = listing_sales_overall['ct_count_listings']/listing_sales_overall['total_households']
	# display(listing_sales_overall.head())

	# merge the overall sales/listings statistics to the census-tract features dataframe 
	df_merged = pd.merge(df_merged, listing_sales_overall[['sales_per_households', 'listings_per_households']], left_on='month', right_index=True)
	df_merged['ct_listings_per_households'] = df_merged['ct_count_listings']/df_merged['total_households']
	df_merged['ct_sales_per_households'] = df_merged['ct_count_sales']/df_merged['total_households']
	df_merged['relative_listings_toBaseline'] = df_merged['ct_listings_per_households']/df_merged['listings_per_households']
	df_merged['relative_sales_toBaseline'] = df_merged['ct_sales_per_households']/df_merged['sales_per_households']
	# df_merged.head()

	# check data for each census tract
	# some census tract only has <10 data points 
	droppped_ct = df_merged.groupby('ct_key').count()['ct_count_sales'].reset_index(name='count')
	droppped_ct = droppped_ct[droppped_ct['count']<10]['ct_key'].values

	# drop census tract with too few data (only 1/2 months of available data)
	df_merged = df_merged[~df_merged['ct_key'].isin(droppped_ct)]

	return df_merged

def preprocess_train_test_data(df_merged, target_column='relative_listings_toBaseline'):
	# number of unique census tracts in dataset
	percent_unique = np.ceil(df_merged.ct_key.nunique()/len(df_merged)*1000)/1000. #round up 3 d.p.

	redundant_columns = ['homeownership_rate', 'family_households_fraction']
	X = df_merged.drop( ['ct_key', 'pop_density', 'census_population', 'month','ct_count_listings', 'ct_count_sales', 'sales_per_households', 'listings_per_households', 
	                     'ct_listings_per_households', 'ct_sales_per_households', 'relative_listings_toBaseline','relative_sales_toBaseline'] + redundant_columns
	                   , axis=1)
	X.columns = X.columns.str.replace(">","greaterthan").str.replace("<","lessthan")

	y = df_merged[target_column].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_unique, random_state=100, stratify = df_merged.ct_key)

	scaler = MinMaxScaler()
	X_train_normalized = scaler.fit_transform(X_train)
	X_train_normalized = pd.DataFrame(X_train_normalized, index=X_train.index, columns=X_train.columns)

	X_test_normalized = scaler.transform(X_test)
	X_test_normalized = pd.DataFrame(X_test_normalized, index=X_test.index, columns=X_test.columns)

	return X_train_normalized, X_test_normalized, y_train, y_test


def plot_contribution(idx, ax, model_top_features, predictions_contrib, X_test):
	predictions_contrib_sample =  predictions_contrib[idx].flatten()
	intercept = predictions_contrib_sample[-1]
	features = predictions_contrib_sample[:-1]
	# sorted_idx = np.argsort(abs(features))[::-1][:30]
	sorted_idx = model_top_features
	sorted_features = features[sorted_idx]

	# intercept bar
	ax.bar(0, intercept, bottom=0., width=0.9, color="green" if intercept> 0 else "red", edgecolor='black')
	this_bottom = intercept
	# all variable bars
	for i in range(len(sorted_features)):
	  next_bottom = this_bottom + sorted_features[i]
	  ax.bar(i+1, next_bottom-this_bottom, bottom=this_bottom, width=0.9,
	          color="green" if next_bottom>this_bottom else "red", edgecolor='black')
	  this_bottom = next_bottom

	# horizontal line for final prediction
	final_pred = predictions_contrib_sample.sum()
	label = f"final prediction = {final_pred:.3f}"
	ax.axhline(final_pred, c='k', linestyle='--', label=label)

	labels = ["intercept"]+list(X_test.columns[sorted_idx])
	ax.set_xticks(np.arange(len(sorted_features)+1))
	ax.set_xticklabels(labels, rotation=90)
	ax.legend()







