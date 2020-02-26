import pickle
import sys
import scipy.io as sio
import copy

sys.path.append('/Users/krishnadave/Desktop/DPGP/')

# Pickle to Python
#file_name = 'a_mixture_model_ARGO_train4'

file_name = 'ARGO_final_DPGP_train4_alpha_1'

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
		
		params = dict()
		
		params["ux"] = data.b[each_z].ux
		params["uy"] = data.b[each_z].uy
		params["sigmax"] = data.b[each_z].sigmax
		params["sigmay"] = data.b[each_z].sigmay
		params["sigman"] = data.b[each_z].sigman
		params["wx"] = data.b[each_z].wx
		params["wy"] = data.b[each_z].wy

		# The following values (params from Frames) are inputs into the covariance function
		# which are defined by the above parameters and return a covariance matrix
		# dependent upon the size of the params from Frames.
		params["x"] = data.frames[each_z].x[0]
#		params["y"] = data.frames[each_z].y[0]
		params["vx"] = data.frames[each_z].vx[0]
		params["vy"] = data.frames[each_z].vy[0]
		'''
		params = []
		params.append(data.b[each_z].ux)
		params.append(data.b[each_z].uy)
		params.append(data.b[each_z].sigmax)
		params.append(data.b[each_z].sigmay)
		params.append(data.b[each_z].sigman)
		params.append(data.b[each_z].wx)
		params.append(data.b[each_z].wy)
		params.append(data.frames[each_z].x[0])
		params.append(data.frames[each_z].y[0])
		params.append(data.frames[each_z].vx[0])
		params.append(data.frames[each_z].vy[0])
		'''
		all_features.append(params)
		

	#print(all_features[0])
	#print(all_features[1])
	#print(len(all_features))

	# Save the features array in matlaba
	# The difference in frames is 1
	delta = 4
	X = copy.copy(all_features[:-delta]) # Historical trajectory parameters

	# Here we may need to create a likelihood for the pattern being predicted
	# based on the features predicted (hat values)
	Y = copy.copy(all_features[delta:])  # Future Trajectory Parameters

	print(len(X), len(Y))
	
	sio.savemat('all_features', {'1': all_features})

m_array1 = sio.loadmat('all_features')
column_name = list(m_array1.keys())[-1]
print(list(m_array1[column_name][0][0]))

# Reference
# https://www.datacamp.com/community/tutorials/pickle-python-tutorial
