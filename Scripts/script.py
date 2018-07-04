import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import xgboost as xgb
import argparse

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Initialize dataframes

# properties_df=pd.DataFrame()
# transaction_df=pd.DataFrame()
# train_df = pd.DataFrame()
# x_train = pd.DataFrame()
# y_train = pd.DataFrame()
# x_test = pd.DataFrame()
# xgb_pred = pd.DataFrame()
# y_mean = 0


def read_data(path):
	try:
		print("Reading data ...")
		# global properties_df, transaction_df,train_df
		properties_df = pd.read_csv(path + "properties_2016.csv", low_memory=False)
		transaction_df = pd.read_csv(path + "train_2016_v2.csv")

		return properties_df, transaction_df
		# train_df = transaction_df.merge(properties_df, how='left', on='parcelid')
		# print "Complete!!"
	except:
		 print("Unexpected error:", sys.exc_info()[0])
		 exit()


def process_data(properties_df, transaction_df):
	# x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
	# y_train = df_train['logerror'].values
	print("Processing data ...")
	# global x_test, x_train, y_train, y_mean, train_df
	for c in properties_df.columns:
		properties_df[c]=properties_df[c].fillna(-1)
		if properties_df[c].dtype == 'object':
			lbl = LabelEncoder()
			lbl.fit(list(properties_df[c].values))
			properties_df[c] = lbl.transform(list(properties_df[c].values))
	train_df = transaction_df.merge(properties_df, how='left', on='parcelid')
	# train_df.to_csv("temp_file.csv")
	x_test = properties_df.drop(['parcelid','propertyzoningdesc', 'propertycountylandusecode'], axis=1)
	train_df = remove_outliers(train_df)
	x_train = train_df.drop(['parcelid', 'logerror','transactiondate','propertyzoningdesc', 'propertycountylandusecode'], axis=1)
	y_train = train_df["logerror"].values.astype(np.float32)

	del train_df, properties_df, transaction_df

	return x_test, x_train, y_train

def read_test_data(path):
	print("Processing data ...")
	properties_df = pd.read_csv(path + "properties_2016.csv", low_memory = False)
	for c in properties_df.columns:
		properties_df[c] = properties_df[c].fillna(-1)
		if properties_df[c].dtype == 'object':
			lbl = LabelEncoder()
			lbl.fit(list(properties_df[c].values))
			properties_df[c] = lbl.transform(list(properties_df[c].values))
	x_test = properties_df.drop(['parcelid','propertyzoningdesc', 'propertycountylandusecode'], axis = 1)

	return x_test, properties_df


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(55, input_dim=55, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_absolute_error', optimizer='adam')
	return model


def deeper_model():
	# create model
	model = Sequential()
	model.add(Dense(55, input_dim=55, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_absolute_error', optimizer='adam')

	return model


def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(125, input_dim=55, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_absolute_error', optimizer='adam')

	return model


def run_XGBOOST(x_train, y_train):
	# global xgb_pred, x_test
	y_mean = np.mean(y_train)
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
	# dtest = xgb.DMatrix(x_test)
	num_boost_rounds = 236
	
	# train model
	print("\nTraining XGBoost ...")
	xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
	print("Saving XGBoost model ...")
	directory="../model"
	if not os.path.exists(directory):
		os.makedirs(directory)
	joblib.dump(xgb_model, "../model/zillow_model.joblib.dat")

	return xgb_model


def make_predictions(xgb_model, x_test):
	dtest = xgb.DMatrix(x_test)
	print("\nPredicting with XGBoost ...")
	xgb_pred = xgb_model.predict(dtest)

	return xgb_pred
	# print("\nXGBoost predictions:")
	# print(pd.DataFrame(xgb_pred).head())


def remove_outliers(train_df):
	print("Removing outliers ...")
	ulimit = np.percentile(train_df.logerror.values, 99)
	llimit = np.percentile(train_df.logerror.values, 1)
	train_df['logerror'].ix[train_df['logerror'] > ulimit] = ulimit
	train_df['logerror'].ix[train_df['logerror'] < llimit] = llimit

	return train_df


def write_results(xgb_pred, properties_df):
	print("\nPreparing results for write ...") 
	y_pred = []
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
	directory="../Output"
	if not os.path.exists(directory):
		os.makedirs(directory)
	path = os.path.join(directory,'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
	output.to_csv(path, index=False)
	print( "\nFinished ..." )


if __name__=="__main__":
	np.random.seed(7)
	file_path = "../Input/"
	model_directory = "../model/zillow_model.joblib.dat"
	parser = argparse.ArgumentParser()
	parser.add_argument("-pt", "--pre_trained", action = "store_true", help = "Use pre-trained model!")
	parser.add_argument("model", help = "Select xg: xgboost or nn: neural network")
	args = parser.parse_args()
	# df_1 = pd.read_csv("../Output/sub20180418_133025.csv", usecols=["201610"])
	# df_1 = np.array(df_1["201610"]) * 0.2
	# df_2 = pd.read_csv("../Output/sub20180418_151341.csv", usecols=["201610"])
	# df_2 = np.array(df_2["201610"]) * 0.7
	# final = df_1 + df_2
	# x_test, properties_df = read_test_data(file_path)
	# write_results(final, properties_df)
	# exit()
	selected_model = args.model
	if selected_model == "xg":
		if args.pre_trained:
			if os.path.exists(model_directory):
				xgb_model = joblib.load(model_directory)
				x_test, properties_df = read_test_data(file_path)
				xgb_pred = make_predictions(xgb_model, x_test)
				write_results(xgb_pred, properties_df)
			else:
				print("Pre-trained model not present in the ../model/ directory")
		else:
			properties_df, transaction_df = read_data(file_path)
			x_test, x_train, y_train = process_data(properties_df, transaction_df)
			xgb_model = run_XGBOOST(x_train, y_train)
			xgb_pred = make_predictions(xgb_model, x_test)
			write_results(xgb_pred, properties_df)
	elif selected_model == "nn":
		properties_df, transaction_df = read_data(file_path)
		x_test, x_train, y_train = process_data(properties_df, transaction_df)
		input_dim = x_train.shape[1]
		print("Input dimension: ", input_dim)
		# model = baseline_model(input_dim)
		# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
		print("Using early stopping ...")
		earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
		# model.fit(X, y, batch_size=128, nb_epoch=100, verbose=1, callbacks=[earlyStopping], validation_split=0.0, validation_data=None, shuffle=True, show_accuracy=False, class_weight=None, sample_weight=None)
		estimators = []
		# estimators.append(('standardize', StandardScaler()))
		estimators.append(('mlp', KerasRegressor(build_fn=wider_model, validation_split=0.33, epochs=50, batch_size=5, verbose=1, callbacks=[earlyStopping])))
		pipeline = Pipeline(estimators)
		pipeline = pipeline.fit(x_train, y_train)
		print("Making neural network predictions ...")
		nn_pred = pipeline.predict(x_test)
		write_results(nn_pred, properties_df)
		# kfold = KFold(n_splits=10, random_state=7)
		# results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
		# print("Neural network: %.2f (%.2f) MSE" % (results.mean(), results.std()))



