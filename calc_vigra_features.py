import vigra
import h5py as h5
import numpy as np
import time
from scipy.ndimage.filters import gaussian_filter

def vigra_gauss(data):
  img = vigra.Image(data)
  sigmas = [0.7, 1.0, 1.6, 3.5, 5, 10]
  out = []
  for sigma in sigmas:
    out.append(np.asarray(vigra.filters.gaussianSmoothing(img, sigma)))

  return out

def vigra_gaussfast(data,kernels):
  img = vigra.Image(data)
  out = []
  for k in kernels:
    tmp = vigra.filters.convolveOneDimension(img,0,k)
    out.append(vigra.filters.convolveOneDimension(tmp,1,k))
  return out

def scipy_gauss(data):
  sigmas = [0.7, 1.0, 1.6, 3.5, 5, 10]
  out = []
  for sigma in sigmas:
    out.append(gaussian_filter(data,sigma,truncate=4.0))

  return out

def create_image_data():
  data = [np.random.random((1024,1024)) for i in range(50)]
  return data

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
#  f = h5.File('data_droso.h5')
#  for s in ['slice1', 'slice2']:
#    f[s + '_fvec'] = calc_features(f[s][:])
#  f.close()
  print "Data creation"
  start = time.clock()
  data = create_image_data()
  end = time.clock()
  print "Time:",end-start

  print "Fast convolution"
  start = time.clock()
  sigmas = [0.7, 1.0, 1.6, 3.5, 5, 10]
  kernels = [vigra.filters.Kernel1D() for i in range(len(sigmas))]
  for k,s in zip(kernels,sigmas):
    k.initGaussian(s)
  for d in data:
    vigra_gaussfast(d,kernels)
  end = time.clock()
  print "Time:",end-start

  print "Slow convolution"
  start = time.clock()
  for d in data:
    vigra_gauss(d)
  end = time.clock()
  print "Time:",end-start

  # compare data
  for v,s in zip(vout,sout):
    print np.linalg.norm(v-s)

### compare scipy and vigra
#  for d in data:
#    start = time.clock()
#    vout = vigra_gauss(d)
#    end = time.clock()
#    print "Vigra time:",end-start
#    start = time.clock()
#    sout = scipy_gauss(d)
#    end = time.clock()
#    print "Scipy time:",end-start
#
#    # compare data
#    for v,s in zip(vout,sout):
#      print np.linalg.norm(v-s)
