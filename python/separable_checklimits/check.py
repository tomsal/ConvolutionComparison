#!/usr/bin/env python

import argparse
import time
import numpy as np
import ctypes as ct
import arrayfire as af
import vigra

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

def af_gaussian2D(s):
  csize = ct.c_int(int(3*s)*2+1)
  csigma = ct.c_double(s)
  d_k = af.Array()
  af.safe_call(af.backend.get().af_gaussian_kernel(ct.pointer(d_k.arr),csize,csize,csigma,csigma))
  return d_k

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run convolution comparison measurements.')
  parser.add_argument("--error", help="Show only failed attempts/errors. Default: false",action="store_true")
  parser.add_argument("--mode", help="Mode of convolution. Common 2D (2D) or separable 2D (sep). Default: 2D",default="2D",type=str)
  parser.add_argument("--device", help="Select GPU device number. Default: 0",default=0,type=int)

  args = parser.parse_args()

  print "--- Parameters ---"
  print "\tError:",args.error
  print "\tMode:",args.mode
  print "\tDevice:",args.device

  af.set_device(args.device)
  af.info()

  sizes = [256,512,1024,2048,3072,4096,5120]
  sigmas = [0.7,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

  for size in sizes:
    afimg = af.randu(size,size)
    for sigma in sigmas:
      try:
	if args.mode == "2D":
	  afk = af_gaussian2D(sigma)
	  afres = af.convolve(afk,afimg)
	if args.mode == "sep":
	  afk = af_gaussianDerivative1D(sigma,0)
	  afres = af.convolve2_separable(afk, af.transpose(afk), afimg)
	if not args.error:
	  print "OK: size =",size,"-- sigma =",sigma,"-- kernel_size =",2*int(3*sigma)+1
      except Exception as e:
	print "ERROR:",e," # size =",size,"-- sigma =",sigma,"-- kernel_size =",2*int(3*sigma)+1

