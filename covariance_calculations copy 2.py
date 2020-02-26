##########################################
# Krishna Dave
##########################################

# Libraries 
import pickle
import sys
import scipy.io as sio
import copy
from numpy import array
from numpy import hstack
import numpy as np
from frame import Frame
import matplotlib
from keras.layers import RepeatVector
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
sys.path.append('/Users/krishnadave/Desktop/DPGP/')

##############################
# Files to work with
file_name1 = 'argo_MixtureModel_2600_2640_1210_1250'
file_name2 = 'argo_MixtureModel_2740_2780_1330_1370'
file_name3 = 'argo_MixtureModel_2780_2810_1360_1390'
file_name4 = 'argo_MixtureModel_2570_2600_1180_1210'
file_name5 = 'argo_MixtureModel_2600_2640_1210_1250'
file_name6 = 'argo_MixtureModel_2640_2670_1240_1270'
file_name7 = 'argo_MixtureModel_2670_2710_1270_1310'
file_name8 = 'argo_MixtureModel_2710_2740_1300_1330'

##############################

def read_dataset(file_name):
	with open(file_name, 'rb') as infile:
		data = pickle.load(infile)
		infile.close()
		return data

def prepare_covMat_data(file_name1):

	read_data = read_dataset(file_name1)
	motion_patterns = read_data.b

	# need to write a helper function for this
	xmin, xmax, ymin, ymax = get_minmax(file_name1)

	WX, WY, ux_pos, uy_pos = covariance_matrix_calc(read_data, motion_patterns, xmin, xmax, ymin, ymax)

	########################
	# Normalizing the matrix
	########################
	WX = np.array(WX)
	WY = np.array(WY)
	ux_pos = np.array(ux_pos)
	uy_pos = np.array(uy_pos)

	WX_mean = np.mean(WX)
	WY_mean = np.mean(WY)
	ux_pos_mean = np.mean(ux_pos)
	uy_pos_mean = np.mean(uy_pos)

	WX_std = np.std(WX)
	WY_std = np.std(WY)
	ux_pos_std = np.std(ux_pos)
	uy_pos_std = np.std(uy_pos)

	WX = (WX-WX_mean)/WX_std 
	WY = (WY-WY_mean)/WY_std
	ux_pos = (ux_pos-ux_pos_mean)/ux_pos_std
	uy_pos = (uy_pos-uy_pos_mean)/uy_pos_std
	########################

	# Putting the covariance matrix in to a new matrix
	new_mat = []
	for i in range(0, len(WX)):
		row = [WX[i][0], WY[i][0], ux_pos[i], uy_pos[i]]
		new_mat.append(row)

	return new_mat


def covariance_matrix_calc(mix_model, motion_patterns, xmin, xmax, ymin, ymax):
	# Takes in the dictionary patt_frame_dict for pattern - frames (key - value pairs) 
	#                         motion patterns  (list with motion pattern info)	
	# Returns the covariance matrix calculated for each pattern based on their corresponding frames
 
	# would the mix model consist of frames for all patterns, or all data?
	
	# The aim of the pattern frame dictionary
	# for each motion pattern, how do I calculate covariance matrix based on the corresponding frames
	for i in range(0, len(motion_patterns)):
		# for pattern in motion_patterns:
		pattern_num = i
		#print("mixmodel.partition", mix_model.partition) # number of frames assigned to the pattern
		pattern_num = np.argmax(np.array(mix_model.partition))
		rate = np.array(mix_model.partition)[i]/mix_model.n
		frame_pattern_ink = mix_model.frame_ink(pattern_num, 0, True)
		# construct mesh frame
		x = np.linspace(xmin, xmax, 31)
		y = np.linspace(ymin, ymax, 31)
		[WX,WY] = np.meshgrid(x, y)
		WX = np.reshape(WX, (-1, 1))
		WY = np.reshape(WY, (-1, 1))
		frame_field = Frame(WX.ravel(), WY.ravel(), np.zeros(len(WX)), np.zeros(len(WX)))
		#get posterior
		ux_pos, uy_pos, covx_pos, covy_pos = mix_model.b[pattern_num].GP_posterior(frame_field, frame_pattern_ink, True)
	return [WX, WY, ux_pos, uy_pos]

def split_sequences(sequences, n_steps_in, n_steps_out):
	# split a multivariate sequence into samples
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
		#print("X.shape, Y.shape", X, y)
	return array(X), array(y)

def split_train_test(array_x, array_y, train_ratio, test_ratio):
	x_len = len(array_x)
	train_num = round(x_len*train_ratio)
	test_num = x_len - train_num

	# Slicing x_train, y_train, x_test, y_test
	x_train = array_x[:train_num]
	x_test = array_x[train_num:]
	y_train = array_y[:train_num]
	y_test = array_y[train_num:]

	return x_train, y_train, x_test, y_test

# Deprecated
def multi_step_lstm(vec_cov_matrix, n_steps_in, n_steps_out):
	# Input is vectorized 1D covariance matrix
	# Predict next 1 or more values based on the first 3-5 values etc...?
	# This would become single input, single output

	# choose a number of time steps
	# split into samples
	X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
	# reshape from [samples, timesteps] into [samples, timesteps, features]
	n_features = 1
	X = X.reshape((X.shape[0], X.shape[1], n_features))
	# define model
	model = Sequential()
	model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
	model.add(LSTM(100, activation='relu'))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X, y, epochs=50, verbose=0)
	# demonstrate prediction
	#x_input = array([70, 80, 90])
	x_input = x_input.reshape((1, n_steps_in, n_features))
	yhat = model.predict(x_input, verbose=0)
	#print(yhat)
	return y_hat

# Helper for LSTM
def get_minmax(file_name):
	splits = file_name.split("_")
	xmin = int(splits[-4])
	xmax = int(splits[-3])
	ymin = int(splits[-2])
	ymax = int(splits[-1])
	return xmin, xmax, ymin, ymax

def lstm(X, y, n_steps_in, n_steps_out):
	
	n_features = X.shape[2]
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
	model.add(RepeatVector(n_steps_out))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(n_features)))
	model.compile(optimizer='adam', loss='mse')
	history = model.fit(X, y, epochs=300, verbose=0)
	return model, history

# Error Functions
def error(model, array_x, array_y):
	y_hat = []
	error_rates  = []
	n_features = array_x.shape[2]
	for i in range(0, len(array_x)):
		x_input = array_x[i]
		x_input = x_input.reshape((1, n_steps_in, n_features))
		yhat = model.predict(x_input, verbose=0)
		y_hat.append(yhat)
		ytar = array_y[i]
		error_rates.append(rmse(yhat, ytar))
		#print("yhat", yhat, "ytar", ytar)
		#break
	return(y_hat, error_rates)

def rmse(predictions, targets):
    differences = predictions - targets
    # print("difference", differences)
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

if __name__ == "__main__":
	cov_matrix = prepare_covMat_data(file_name1)
	#norm_matrix = normalize(cov_matrix)
	dummy_matrix = [[ 10.0, 15.0, 25.0],
		 [ 20.0, 25.0, 45.0],
		 [ 30.0, 35.0, 1000.0], 
		 [ 40.0, 45.0, 85.0],
		 [ 50.0, 55.0, 105.0],
		 [ 60.0, 65.0, 125.0],
		 [ 70.0, 75.0, 145.0],
		 [ 80.0, 85.0, 165.0],
		 [ 90.0, 95.0, 185.0]]

	#dummy_matrix = tf.cast(dummy_matrix, tf.float32)

	# print(len(dummy_matrix), len(dummy_matrix[0]))
	
	n_steps_in, n_steps_out = 5, 2
	array_x, array_y = split_sequences(cov_matrix, n_steps_in, n_steps_out)
	#print("array_x", array_x)
	#print("array_y" , array_y)
	print("len array x", len(array_x))
	train_ratio = .7
	test_ratio  = .3
	x_train, y_train, x_test, y_test = split_train_test(array_x, array_y, train_ratio, test_ratio)

	# Training the model
	trained_model, train_hist = lstm(x_train, y_train, n_steps_in, n_steps_out)

	plot_train_history(train_hist, "Training loss")

	# Training Error
	# predict y_train_hat using x_train and model 
	#ytrain_hat = predict(model, x_train)
	# Calculate rmse between y_train_hat and y_train
	#rmse = rmse(ytrain_hat, y_train)
	# for each case and total rmse
	ytrain_hat, train_error = error(trained_model, x_train, y_train)
	
	ytest_hat, test_error = error(trained_model, x_test, y_test)

	'''
	print("train_error", train_error)
	print("test_error", test_error)
	print("ytrain_hat", ytrain_hat)
	print("ytest_hat", ytest_hat)
	print("y_train", y_train)
	print("y_test", y_test)
	'''

	import matplotlib.pyplot as plt 
	import numpy as np
	import pandas as pd

	time = list(range(1, len(test_error)+1))
	print(len(time))
	print("len train error", len(test_error))

	df=pd.DataFrame({'x': time, 'test_error': test_error})
	plt.plot('x', 'test_error', data = df, marker = '', color = "olive", linewidth = 2)
	#plt.plot('x', 'y2', data = df, marker = '', color = "blue", linewidth = 2)
	plt.legend()
	plt.show()

	#plt.plot(time, train_error)
	#plt.plot(time, test_error)
	#return(y_hat, error_rates)

	# Test Error
	# predict y_test_hat using x_test and model 
	# Calculate rmse between y_test_hat and y_test
	# for each case and total rmse
	

	# Draw outputs using Matplotlib

	# How do outputs look like for time-sequence data?
	# train error vs. time sequence
	# test_error vs. time_sequence
	# error vs. epoch will have to do it while training the model, so look into inbuilt functions)



