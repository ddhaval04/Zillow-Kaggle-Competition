import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import xgboost as xgb
import os
import warnings
warnings.filterwarnings("ignore")

# Initialize dataframes

properties_df=pd.DataFrame()
transaction_df=pd.DataFrame()
train_df = pd.DataFrame()
x_train = pd.DataFrame()
y_train = pd.DataFrame()
x_test = pd.DataFrame()
xgb_pred = pd.DataFrame()
y_mean = 0
def read_data(path):
	try:
		print "Reading data ..."
		global properties_df, transaction_df,train_df
		
		properties_df = pd.read_csv(path+"/properties_2016.csv", low_memory=False)
		transaction_df = pd.read_csv(path+"/train_2016_v2.csv")
		# train_df = transaction_df.merge(properties_df, how='left', on='parcelid')
		# print "Complete!!"
	except:
		 print "Unexpected error:", sys.exc_info()[0]

def process_data():
	# x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
	# y_train = df_train['logerror'].values
	print "Processing data for XGBoost ..."
	global x_test, x_train, y_train, y_mean, train_df
	for c in properties_df.columns:
		properties_df[c]=properties_df[c].fillna(-1)
		if properties_df[c].dtype == 'object':
			lbl = LabelEncoder()
			lbl.fit(list(properties_df[c].values))
			properties_df[c] = lbl.transform(list(properties_df[c].values))
	train_df = transaction_df.merge(properties_df, how='left', on='parcelid')
	# train_df.to_csv("temp_file.csv")
	x_test = properties_df.drop(['parcelid','propertyzoningdesc', 'propertycountylandusecode'], axis=1)
	remove_outliers()
	x_train = train_df.drop(['parcelid', 'logerror','transactiondate','propertyzoningdesc', 'propertycountylandusecode'], axis=1)
	y_train = train_df["logerror"].values.astype(np.float32)
	y_mean = np.mean(y_train)	
	print(x_train.shape, y_train.shape)
	

def run_XGBOOST():
	global xgb_pred, x_test
	xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,   
    'alpha': 0.3995, 
    'base_score': y_mean,
    'silent': 1
	}
	dtrain = xgb.DMatrix(x_train, y_train)
	dtest = xgb.DMatrix(x_test)
	num_boost_rounds = 236
	
	# train model
	print "\nTraining XGBoost ..."
	xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
	print "\nPredicting with XGBoost ..."
	xgb_pred = xgb_model.predict(dtest)
	print "\nXGBoost predictions:" 
	print pd.DataFrame(xgb_pred).head()
	
def remove_outliers():
	print "Removing outliers ..."
	ulimit = np.percentile(train_df.logerror.values, 99)
	llimit = np.percentile(train_df.logerror.values, 1)
	train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit
	train_df['logerror'].ix[train_df['logerror']<llimit] = llimit
	
def write_results():
	print "\nPreparing results for write ..." 
	y_pred=[]
	for i,predict in enumerate(xgb_pred):
		y_pred.append(str(round(predict,4)))
	y_pred = np.array(y_pred)
	output = pd.DataFrame({'ParcelId': properties_df['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
	cols = output.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	output = output[cols]
	
	print( "\nWriting results to disk ..." )
	directory=r"E:\Dhaval\Data Science\Kaggle\Zillow\Output"
	if not os.path.exists(directory):
		os.makedirs(directory)
	path = os.path.join(directory,'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
	output.to_csv(path, index=False)
	print( "\nFinished ..." )
	
	
if __name__=="__main__":
	file_path = r"E:\Dhaval\Data Science\Kaggle\Zillow\Input"
	read_data(file_path)
	process_data()
	run_XGBOOST()
	write_results()