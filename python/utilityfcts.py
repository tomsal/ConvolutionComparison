#!/usr/bin/env python

import vigra
import arrayfire as af
import ctypes as ct

# print vigra kernel
def pk(k):
  for i in range(k.left(),k.right()+1):
    print k[i]

def af_gaussianDerivative1D(s,order):
  radius = int(round(3*s + 0.5*order))
  size = radius*2+1
  csize = ct.c_int(size)
  csigma = ct.c_double(s)
  d_k = af.Array()
  af.safe_call(af.backend.get().af_gaussian_kernel(ct.pointer(d_k.arr),csize,1,csigma,0))
  if order == 1:
    afx=af.range(size)-radius
    return -afx/s/s*d_k
  if order == 2:
    afx=((af.range(size)-radius)**2-s*s)
    return afx/(s**4)*d_k
  return d_k

def af_hessianOfGaussian(img,smooth_k,deriv1_k,deriv2_k):
  resx = af.convolve2_separable(smooth_k, af.transpose(deriv2_k), img)
  resy = af.convolve2_separable(deriv2_k, af.transpose(smooth_k), img)
  resxy = af.convolve2_separable(deriv1_k, af.transpose(deriv1_k), img)
  return resx,resy,resxy
