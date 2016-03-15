#!/usr/bin/env python

import os

sigmas = [0.7,1.0,1.6,3.5,5.0,10.0]
sizes = [256,512,1024,1500,2048,2500,3072,3500]
N = "20"

os.system("./a.out --N 0 --raw > data/data.dat")
for size in sizes:
  size = str(size)
  for sigma in sigmas: 
    sigma = str(sigma)
    print "./a.out --sigma "+sigma+" --size "+size+" --N "+N+" --raw >> data/data.dat"
    os.system("./a.out --sigma "+sigma+" --size "+size+" --N "+N+" --raw >> data/data.dat")
