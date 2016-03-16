#!/usr/bin/env python

import os
import argparse

parser = argparse.ArgumentParser(description='Run several convolution measurements.')
parser.add_argument("--N", help="Amount of measurements taken. Default: 10",default=10,type=int)
parser.add_argument("--device", help="GPU device used for measurements. Default: 0",default=0,type=int)
parser.add_argument("--cputhreads", help="Amount threads used for vigra convolution. Default: 1",default=1,type=int)
args = parser.parse_args()

# run parameters
sigmas = [0.7,1.0,1.6,3.5,5.0,10.0]
sizes = [256,512,1024,1500,2048,2500,3072,3500]
N = str(args.N)
device = str(args.device)
cputhreads = str(args.cputhreads)

os.system("./a.out --N 0 --raw > data/data.dat")
for size in sizes:
  size = str(size)
  for sigma in sigmas: 
    sigma = str(sigma)
    print "./a.out --cputhreads "+cputhreads+" --device "+device+" --sigma "+sigma+" --size "+size+" --N "+N+" --raw >> data/data.dat"
    os.system("./a.out --cputhreads "+cputhreads+" --device "+device+" --sigma "+sigma+" --size "+size+" --N "+N+" --raw >> data/data.dat")
