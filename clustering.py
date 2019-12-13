from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def clustering(matrix, cluster_function, n=5, coords=False):
	x = matrix.shape[0]
	y = matrix.shape[1]
	z = matrix.shape[2]
	n_components = matrix.shape[3]

	if coords:
		matrix = add_space_features(matrix)
		matrix = matrix.reshape([x*y*z, n_components+3])
	else:
		matrix = matrix.reshape([x*y*z, n_components])

	sc = StandardScaler()
	matrix = sc.fit_transform(matrix)

	clusterer = cluster_function(n_clusters=n)
	# clusterer = cluster_function()
	result = clusterer.fit_predict(matrix)
	# print(result.shape)

	result = result.reshape([x, y, z, 1])	

	# plt.imshow(result[:,:,10, 0])
	# plt.show()
	# print('Clustering acomplished!')
	return result

def add_space_features(matrix):
	x = matrix.shape[0]
	y = matrix.shape[1]
	z = matrix.shape[2]
	features = matrix.shape[3]
	new_matrix = np.zeros([x, y, z, features+3])
	for i in range(x):
		for j in range(y):
			for k in range(z):
				for f in range(features):
					new_matrix[i,j,k,f] = matrix[i,j,k,f]
				new_matrix[i,j,k,features] = i
				new_matrix[i,j,k,features+1] = j
				new_matrix[i,j,k,features+2] = k

	return new_matrix