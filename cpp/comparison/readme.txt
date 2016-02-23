# optional load nvidia driver
option
nvidia-smi

# add conda libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda2/envs/ilastik-devel/lib

# compile
g++ main_new.cpp -laf -lafcuda `vigra-config --cppflags --libs` -O2 -mtune=native -lboost_program_options -lboost_system -lboost_timer
