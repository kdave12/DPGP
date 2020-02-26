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
file_name1 = 'argo_MixtureModel_2740_2780_1330_1370'
file_name2 = 'argo_MixtureModel_2780_2810_1360_1390'
file_name3 = 'argo_MixtureModel_2570_2600_1180_1210'
file_name4 = 'argo_MixtureModel_2600_2640_1210_1250'
file_name5 = 'argo_MixtureModel_2640_2670_1240_1270'
file_name6 = 'argo_MixtureModel_2670_2710_1270_1310'
file_name7 = 'argo_MixtureModel_2710_2740_1300_1330'

##############################

def read_dataset(file_name):
	with open(file_name, 'rb') as infile:
		data = pickle.load(infile)
		infile.close()
		return data

def makedict_pattframe(data):
	# Each frame has an associated pattern.
	# This function makes a dictionary such that: {pattern1: frames, pattern2: frames ...}

	# Each frame has an associated pattern. Make a dictionary such that: 
	# {patterns: frames}
	
	#print("data type", type(data))
	#for each_frame in all_features:
	
	# Total number of motion patterns
	# print("len of data.b", len(data.b))
	# print("data.b", data.b)

	#data.z - indexes pointing to b
	# print("len of data.z", len(data.z))
	# print("data.z", data.z)

	patt_frame_dict = dict()

	for i in range(0, len(data.z)):
		if data.z[i] not in patt_frame_dict:
			patt_frame_dict[data.z[i]] = [i]
		else:
			patt_frame_dict[data.z[i]].append(i)

	# print(patt_frame_dict)
	return patt_frame_dict

def covariance_matrix_calc(mix_model, patt_frame_dict, motion_patterns, xmin, xmax, ymin, ymax):
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

def vectorize_covar_matrix(cov_mat):
	# after having an n by n square covariance matrix, you can vectorize it into a long covariance matrix
	vec_cov_mat = np.ravel(cov_mat)
	return vec_cov_mat

def split_sequences(sequence, n_steps_in, n_steps_out):
	# split a univariate sequence into samples
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def multi_step_lstm(vec_cov_matrix):
	# Input is vectorized 1D covariance matrix
	# Predict next 1 or more values based on the first 3-5 values etc...?
	# This would become single input, single output

	# choose a number of time steps
	n_steps_in, n_steps_out = 3, 2
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

def lstm():
	read_data = read_dataset(file_name1)
	motion_patterns = read_data.b # data.b
	patt_frame_dict = makedict_pattframe(read_data)
	xmin = 2600
	xmax = 2640
	ymin = 1210
	ymax = 1250
	res = covariance_matrix_calc(read_data, patt_frame_dict, motion_patterns, xmin, xmax, ymin, ymax)
	WX = res[0]
	WY = res[1]
	ux_pos = res[2]
	uy_pos = res[3]

	print(type(WX), type(WY), type(ux_pos), type(uy_pos))

	new_mat = []
	for i in range(0, len(res[0])):
		row = [WX[i], WY[i], ux_pos[i], uy_pos[i]]
		new_mat.append(row)

	array_x, array_y = split_sequences(new_mat, 5, 2)

	print("array_x", len(array_x[0]))
	print("array_y", len(array_y[0]))
	print(len(array_x[0]), len(array_y[0]))

	n_steps_in = 5
	n_steps_out = 2
	X, y = split_sequences(np.asarray(new_mat), n_steps_in, n_steps_out)
	# the dataset knows the number of features, e.g. 2
	n_features = X.shape[2]
	# define model
	model = Sequential()

	model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
	model.add(RepeatVector(n_steps_out))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(n_features)))
	model.compile(optimizer='adam', loss='mse')
	model.fit(X, y, epochs=300, verbose=0)

def training_error():
	error_rates  = []
	for i in range(0, len(array_x)):
		x_input = array_x[i]
	x_input = x_input.reshape((1, n_steps_in, n_features))
	yhat = model.predict(x_input, verbose=0)
	ytar = array_y[i]
	error_rates.append(rmse(yhat, ytar))
	return error_rates

def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

def get_error_rate():
	error_rates = training_error()
	return error_rates



