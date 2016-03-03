# optional load nvidia driver
option
nvidia-smi

# add conda libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda2/envs/ilastik-devel/lib

# compile
g++ main.cpp -O2 -mtune=native `vigra-config --cppflags --libs` -lboost_program_options -lboost_system -lboost_timer -laf -lafcuda

# use run.py batch measurements
python run.py

# use gnuplot for plotting the data
gnuplot plotscript
