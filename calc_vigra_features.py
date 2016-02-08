import vigra
import h5py as h5
import numpy as np
import time
import arrayfire as af
import ctypes as ct
import cv2
from scipy.ndimage.filters import gaussian_filter


def check(name,epsilon):
  if epsilon > 1e-4:
    print "### Not matching for:",name
    print "### Error:",epsilon

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
  return out

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
  return out

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
  return out

def cv2_gauss(data,sigmas):
  out = []
  sigmasizes = ((3*sigmas+0.5).astype(int)*2+1)
  for d in data:
    for sigma,sigmasize in zip(sigmas,sigmasizes):
      out.append(cv2.GaussianBlur(d,(sigmasize,sigmasize),sigma))

  return out

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

def create_image_data(size=1024,N=1):
  data = [np.random.random((size,size)) for i in range(N)]
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
  af.set_device(2)
  af.info()

#  f = h5.File('data_droso.h5')
#  for s in ['slice1', 'slice2']:
#    f[s + '_fvec'] = calc_features(f[s][:])
#  f.close()
  sigmas = np.array([0.7, 1.0, 1.6, 3.5, 5]) # , 10])
  kernels = [vigra.filters.Kernel1D() for i in range(len(sigmas))]
  for k,s in zip(kernels,sigmas):
    k.initGaussian(s)

  npkernels = []
  for k in kernels:
    k1 = np.zeros(k.size())

    for i,j in zip(range(k.size()),range(k.left(), k.size()+1)):
      k1[i] = k[j] 
    npkernels.append(k1)

  # does not work for size > 4000 on my notebook
  for size in [256,1024,2048,3072, 3500,4096,4500,5120,10000]:
    print "--------------------------------"
    start = time.clock()
    data = create_image_data(size,1)
    end = time.clock()
    print "Create data, size:",size,"time:",end-start,"s"

    start = time.clock()
    af0 = af_gauss_sigmas_sep(data,sigmas)
    end = time.clock()
    print "af_gauss_sigmas_sep, time:",end-start,"s"

    start = time.clock()
    af1 = af_gauss_sigmas(data,sigmas)
    end = time.clock()
    print "af_gauss_sigmas\ttime:",end-start,"s"

    start = time.clock()
    af2 = af_gauss(data,npkernels)
    end = time.clock()
    print "af_gauss\ttime:",end-start,"s"

    start = time.clock()
    cv1 = cv2_gauss(data,sigmas)
    end = time.clock()
    print "opencv_gauss\ttime:",end-start,"s"

    start = time.clock()
    vg1 = vigra_gauss(data,sigmas)
    end = time.clock()
    print "vigra_gauss\ttime:",end-start,"s"

    print "Compare data"
    # arrayfire seems todo 0 padding
    # compare data
    for a0,a1,a2,c1,v1 in zip(af0,af1,af2,cv1,vg1):
      check('af0',np.linalg.norm(v1[100:110,100:110]-a0[100:110,100:110]))
      check('af1',np.linalg.norm(v1[100:110,100:110]-a1[100:110,100:110]))
      check('af2',np.linalg.norm(v1[100:110,100:110]-a2[100:110,100:110]))
      check('cv1',np.linalg.norm(v1[100:110,100:110]-c1[100:110,100:110]))
      check('vg1',np.linalg.norm(v1[100:110,100:110]-v1[100:110,100:110]))
