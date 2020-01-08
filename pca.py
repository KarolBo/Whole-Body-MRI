import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reduce_dimensionality(matrix, n_components=3):
	x = matrix.shape[0]
	y = matrix.shape[1]
	z = matrix.shape[2]
	features = matrix.shape[3]
	matrix = matrix.reshape([x*y*z, features])

	print('shape before:', matrix.shape)
	
	sc = StandardScaler()
	train_features = sc.fit_transform(matrix)

	pca = PCA(n_components=n_components)
	train_pca = pca.fit_transform(train_features)
	
	print('shape after:', train_pca.shape)
	print("variance:", sum(pca.explained_variance_ratio_))

	train_pca = train_pca.reshape([x, y, z, n_components])

	return train_pca