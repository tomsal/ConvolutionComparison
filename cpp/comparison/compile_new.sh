#/usr/bin/env bash

g++ main_new.cpp -laf -lafcuda `vigra-config --cppflags --libs` -O2 -mtune=native
g++ main_new.cpp -laf -lafcuda -I/home/tommy/miniconda2/envs/ilastik-devel/include
-L/home/tommy/miniconda2/envs/ilastik-devel/lib -lvigraimpex -ljpeg -lpng -lz -ltiff -lhdf5 -lhdf5_hl -O2 -mtune=native -lboost_program_options
g++ main_new.cpp -laf -lafcuda -I/home/tommy/miniconda2/envs/ilastik-devel/include
-L/home/tommy/miniconda2/envs/ilastik-devel/lib -lvigraimpex -ljpeg -lpng -lz -ltiff -lhdf5 -lhdf5_hl -O2 -mtune=native -lboost_program_options
