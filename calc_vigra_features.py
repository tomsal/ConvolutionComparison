import vigra
import h5py as h5
import numpy as np
import time
import arrayfire as af
import ctypes as ct
from scipy.ndimage.filters import gaussian_filter


def af_gauss_sigmas_sep(data,sigmas):
  d_kernels = []
  for s in sigmas:
    csize = ct.c_int(int(3*s + 0.5)*2+1)
    csigma = ct.c_double(s)
    d_k = af.Array()
    af.safe_call(af.backend.get().af_gaussian_kernel(ct.pointer(d_k.arr),csize,1,csigma,ct.c_double(0.0)))
    d_kernels.append(d_k)
  
  out = []
  for d in data:
    d_img = af.np_to_af_array(d)

    for d_k,i in zip(d_kernels,range(len(d_kernels))):
      res = af.convolve2_separable(d_k, af.transpose(d_k), d_img)
      # create numpy array
      out.append(res.__array__())

def af_gauss_sigmas(data,sigmas):
  d_kernels = []
  for s in sigmas:
    csize = ct.c_int(int(3*s + 0.5)*2+1)
    csigma = ct.c_double(s)
    d_k = af.Array()
    af.safe_call(af.backend.get().af_gaussian_kernel(ct.pointer(d_k.arr),csize,csize,csigma,csigma))
    d_kernels.append(d_k)
  
  out = []
  for d in data:
    d_img = af.np_to_af_array(d)

    for d_k in d_kernels:
      #res = af.convolve2_separable(d_k, af.transpose(d_k), d_img)
      res = af.convolve2(d_img, d_k)
      # create numpy array
      out.append(res.__array__())

def af_gauss(data,kernels):
  d_kernels = []
  for k in kernels:
    d_k = af.np_to_af_array(k)
    d_kernels.append(af.matmul(d_k,af.transpose(d_k)))
  
  ##d_k = af.matmul(af.transpose(d_k),d_k)
  out = []
  for d in data:
    d_img = af.np_to_af_array(d)

    for d_k in d_kernels:
      #res = af.convolve2_separable(d_k, af.transpose(d_k), d_img)
      res = af.convolve2(d_img, d_k)
      # create numpy array
      out.append(res.__array__())

def vigra_gauss(data,sigmas):
  out = []
  for d in data:
    img = vigra.Image(d)
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

def create_image_data(size=1024):
  data = [np.random.random((size,size)) for i in range(10)]
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
  sigmas = [0.7, 1.0, 1.6, 3.5, 5]#, 10]
  kernels = [vigra.filters.Kernel1D() for i in range(len(sigmas))]
  for k,s in zip(kernels,sigmas):
    k.initGaussian(s)

  npkernels = []
  for k in kernels:
    k1 = np.zeros(k.size())

    for i,j in zip(range(k.size()),range(k.left(), k.size()+1)):
      k1[i] = k[j] 
    npkernels.append(k1)

  # does not work for size > 4000
  for size in [256,512,1024,2048,3072,3500,3800]:
    start = time.clock()
    data = create_image_data(size)
    end = time.clock()
    print "Create data, size:",size,"time:",end-start,"s"

    start = time.clock()
    af_gauss_sigmas_sep(data,sigmas)
    end = time.clock()
    print "af_gauss_sigmas_sep, time:",end-start,"s"

    start = time.clock()
    af_gauss_sigmas(data,sigmas)
    end = time.clock()
    print "af_gauss_sigmas, time:",end-start,"s"

    start = time.clock()
    af_gauss(data,npkernels)
    end = time.clock()
    print "af_gauss, time:",end-start,"s"

    start = time.clock()
    vigra_gauss(data,sigmas)
    end = time.clock()
    print "vigra_gauss, time:",end-start,"s"

#  # compare data
#  for v,s in zip(vout,sout):
#    print np.linalg.norm(v-s)

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
