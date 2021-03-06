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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run convolution comparison measurements.')
  parser.add_argument("--N", help="Amount of measurements taken. Default: 10",default=10,type=int)
  parser.add_argument("--sigma", help="Sigma of gaussian filter. Default: 1.0",default=1.0,type=float)
  parser.add_argument("--size", help="Size of squared image. Default: 1024",default=1024,type=int)
  parser.add_argument("--raw", help="Produce raw output for plot data.",action="store_true")
  parser.add_argument("--device", help="Select GPU device number. Default: 0",default=0,type=int)

  args = parser.parse_args()

  if not args.raw:
    print "--- Parameters ---"
    print "\tSize:",args.size
    print "\tN:",args.N
    print "\tSigma:",args.sigma
    print "\tDevice:",args.device

  af.set_device(args.device)

  # create input image
  img = np.random.random((args.size,args.size))

  # create arrayfire gaussiankernel
  start = time.clock()
  afsmk = af_gaussianDerivative1D(args.sigma,0)
  afd1k = af_gaussianDerivative1D(args.sigma,1)
  afd2k = af_gaussianDerivative1D(args.sigma,2)
  end = time.clock()
  af_kernel = end-start

  # storage for times
  af_cpy_hd = np.zeros(args.N)
  af_convolve = np.zeros(args.N)
  af_cpy_dh = np.zeros(args.N)
  vigra_t = np.zeros(args.N)

  if not args.raw:
    print "--- Starting measurements ---"

  for i in range(args.N):
    ## arrayfire measurement
    start = time.clock()
    #---
    afimg = af.np_to_af_array(img)
    af.sync()
    #---
    end = time.clock()
    af_cpy_hd[i] = end-start

    start = time.clock()
    #---
    afresx = af.convolve2_separable(afsmk, af.transpose(afd2k), afimg)
    afresy = af.convolve2_separable(afd2k, af.transpose(afsmk), afimg)
    afresxy = af.convolve2_separable(afd1k, af.transpose(afd1k), afimg)
    af.sync()
    #---
    end = time.clock()
    af_convolve[i] = end-start

    start = time.clock()
    #---
    afresx_h = afresx.__array__()
    afresy_h = afresy.__array__()
    afresxy_h = afresxy.__array__()
    af.sync()
    #---
    end = time.clock()
    af_cpy_dh[i] = end-start

    ## vigra measurement
    start = time.clock()
    #---
    vimg = vigra.Image(img)
    vres = np.asarray(vigra.filters.hessianOfGaussianEigenvalues(vimg, args.sigma))
    #---
    end = time.clock()
    vigra_t[i] = end-start

  if not args.raw:
    print "--- Results ---"
    print "\tAf kernel creation =",af_kernel
    print "\tMean copy h->d =",np.mean(af_cpy_hd),"("+str(np.sqrt(np.var(af_cpy_hd)))+")"
    print "\tMean convolve =",np.mean(af_convolve),"("+str(np.sqrt(np.var(af_convolve)))+")"
    print "\tMean copy d->h =",np.mean(af_cpy_dh),"("+str(np.sqrt(np.var(af_cpy_dh)))+")"
    print "\tMean arrayfire =",np.mean(af_cpy_hd+af_cpy_dh+af_convolve),"("+str(np.sqrt(np.var(af_cpy_hd+af_cpy_dh+af_convolve)))+")"
    print "\tMean vigra =",np.mean(vigra_t),"("+str(np.sqrt(np.var(vigra_t)))+")"

  else: # print raw
    if args.N > 0:
      print str(args.size)+", "+str(args.sigma)+", "+str(af_kernel)+", "+\
	    str(np.mean(af_cpy_hd))+", "+str(np.sqrt(np.var(af_cpy_hd)))+", "+\
	    str(np.mean(af_convolve))+", "+str(np.sqrt(np.var(af_convolve)))+", "+\
	    str(np.mean(af_cpy_dh))+", "+str(np.sqrt(np.var(af_cpy_dh)))+", "+\
	    str(np.mean(af_cpy_hd+af_cpy_dh+af_convolve))+", "+str(np.sqrt(np.var(af_cpy_hd+af_cpy_dh+af_convolve)))+", "+\
	    str(np.mean(vigra_t))+", "+str(np.sqrt(np.var(vigra_t)))
    else:
      print "# size, sigma, af kernel [s], af cpy h->d mean [s], af cpy h->d stddev [s],"+\
	    " af convolve mean [s], af convolve stddev [s],"+\
	    " af cpy d->h mean [s], af cpy d->h stddev [s],"+\
	    " af total mean [s], af total stddev [s],"+\
	    " vigra mean [s], vigra stddev [s]"
 

