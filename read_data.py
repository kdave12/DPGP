import pickle
import sys
import scipy.io as sio
import copy

sys.path.append('/Users/krishnadave/Desktop/DPGP/')

# Pickle to Python
file_name1 = 'argo_MixtureModel_2740_2780_1330_1370'
file_name2 = 'argo_MixtureModel_2780_2810_1360_1390'
file_name3 = 'a_mixture_model_ARGO_all'
file_name4 = 'argo_MixtureModel_2570_2600_1180_1210'
file_name5 = 'argo_MixtureModel_2600_2640_1210_1250'
file_name6 = 'argo_MixtureModel_2640_2670_1240_1270'
file_name7 = 'argo_MixtureModel_2670_2710_1270_1310'
file_name8 = 'argo_MixtureModel_2710_2740_1300_1330'
file_name9 = 'ARGO_final_DPGP_all_alpha_1'
file_name10 = 'ARGO_final_DPGP_train4_alpha_1'
file_name11 = 'a_mixture_model_ARGO_train4'
file_name12 = 'frame_map_range_0_argo_all'

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

			params.append(data.b[each_z].ux)
			params.append(data.b[each_z].uy)
			params.append(data.b[each_z].sigmax)
			params.append(data.b[each_z].sigmay)
			params.append(data.b[each_z].sigman)
			params.append(data.b[each_z].wx)
			params.append(data.b[each_z].wy)

			all_features.append(params)
		
		return(all_features)

#output1 = read_dataset(file_name1)
output = read_dataset(file_name10)

#print(output1 == output2)


print("filename: ", file_name10)
print("type: ", type(output))
print("# of rows: ",  len(output))
print("# of cols: ",  len(output[1]))






# Reference
# https://www.datacamp.com/community/tutorials/pickle-python-tutorial
