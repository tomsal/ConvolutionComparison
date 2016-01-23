import vigra
import h5py as h5
import numpy as np


def calc_features(data):
	img = vigra.Image(data)

	sigmas = [0.7, 1.0, 1.6, 3.5, 5, 10]
	features = np.zeros((8*len(sigmas) + 1, data.shape[0], data.shape[1]))

	features[0,:,:] = np.asarray(vigra.filters.gaussianSmoothing(img, 0.3))

	for i, sigma in enumerate(sigmas):
		stensor = vigra.filters.structureTensorEigenvalues(img, sigma, sigma/2.0)
		hogev = vigra.filters.hessianOfGaussianEigenvalues(img, sigma)

		features[1 + 0*len(sigmas) + i, :, :] = np.asarray(vigra.filters.gaussianSmoothing(img, sigma))
		features[1 + 1*len(sigmas) + i, :, :] = np.asarray(vigra.filters.laplacianOfGaussian(img, sigma))
		features[1 + 2*len(sigmas) + i, :, :] = np.asarray(vigra.filters.gaussianGradientMagnitude(img, sigma))
		features[1 + 3*len(sigmas) + i, :, :] = np.asarray(vigra.filters.gaussianSmoothing(img, sigma) - vigra.filters.gaussianSmoothing(img, 0.66*sigma))
		features[1 + 4*len(sigmas) + i, :, :] = np.asarray(stensor[:,:,0])
		features[1 + 5*len(sigmas) + i, :, :] = np.asarray(stensor[:,:,1])
		features[1 + 6*len(sigmas) + i, :, :] = np.asarray(hogev[:,:,0])
		features[1 + 7*len(sigmas) + i, :, :] = np.asarray(hogev[:,:,1])

	features = features.swapaxes(0, 2).swapaxes(0, 1)
	return features

if __name__ == "__main__":
	f = h5.File('data_droso.h5')
	for s in ['slice1', 'slice2']:
		f[s + '_fvec'] = calc_features(f[s][:])
	f.close()