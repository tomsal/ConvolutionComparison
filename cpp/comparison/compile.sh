#/usr/bin/env bash

g++ main.cpp -laf -lafcuda `vigra-config --cppflags --libs` -O2 -mtune=native
