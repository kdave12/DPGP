import pickle
import sys
import scipy.io as sio
import copy
from numpy import array
from numpy import hstack
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

sys.path.append('/Users/krishnadave/Desktop/DPGP/')

# Pickle to Python
#file_name = 'a_mixture_model_ARGO_train4'

file_name1 = 'ARGO_final_DPGP_train4_alpha_1'

def read_dataset(file_name):
	with open(file_name, 'rb') as infile:
		data = pickle.load(infile)
		infile.close()
		# data is a mixture models class
		# Need to extract the attributes from the class and turn them into 
		# an array, so you could change that to .MAT file and pass into

		# Array of X
		array_x = []
		# Array Y
		array_y = []

		# Total of two motion patterns
		#print(len(data.b))

		# Parameters:
		# Motion Patterns: ux, uy, sigmax, sigamy, sigman, wx, wy
		# Frames: x, y, vx, vy

		all_features = []
		z = data.z # indexes the pointing to b
		for each_z in z: # per frame
			
			#params = dict()
			params = []

			# To calculate the number of unique motion patterns
			#datab_set = len(set(data.b))
			#print("number of motion patterns", datab_set)
			#break 
			params.append(data.b[each_z].ux)
			params.append(data.b[each_z].uy)
			params.append(data.b[each_z].sigmax)
			params.append(data.b[each_z].sigmay)
			params.append(data.b[each_z].sigman)
			params.append(data.b[each_z].wx)
			params.append(data.b[each_z].wy)

			all_features.append(array(params))
		
		print(all_features)
		return(array(all_features))

# want to make split sequence more modular..... TODO 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)-1):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 

## Main 
full_seq = read_dataset(file_name1)
n_steps = 5
x_test_full = array(full_seq[-6:])
X, y = split_sequences(full_seq, n_steps)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=400, verbose=0)
# demonstrate prediction
x_test = x_test_full[:-1]
y_test = x_test_full[-1] # Target
print("x_test", x_test)
print("y_test", y_test)

x_test = x_test.reshape((1, n_steps, n_features))

# Prediction
y_hat = model.predict(x_test, verbose=0)
print("yhat", y_hat)

import numpy as np

pred = np.array(y_hat)
tar = np.array(y_test)

def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

rmse_val = "rms error is: " + str(rmse(pred, tar))

print(rmse_val)

# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/




