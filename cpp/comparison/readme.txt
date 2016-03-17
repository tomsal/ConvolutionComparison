This folder contains the main C++ comparison between arrayfire and vigra. It requires vigra 1.11 rc to run the blockwise multithreaded convolution.
Subfolders:
data - contains data recorded with run.py
plots - contains png files of some plots
plotscript - gnuplot scripts to plot the data in data/

# On my personal notebook.
# optional load nvidia driver
option
nvidia-smi

# add conda/boost libraries in case you get an shared library error when running the program.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda2/envs/ilastik-devel/lib

# compile
# on notebook:
g++ main.cpp -O2 -mtune=native `~/Documents/uni/HCI_HiWi/vigra_local/bin/vigra-config --cppflags --libs` -lboost_program_options -lboost_system -lboost_timer -laf -lafcuda -std=c++11
# on bigheron
g++ main.cpp -O2 -mtune=native `/export/home/thehn_local/vigra/vigra_local/bin/vigra-config --cppflags --libs` -lboost_program_options -lboost_system -lboost_timer -laf -lafcuda -std=c++11

# use run.py batch measurements
python run.py

# use gnuplot for plotting the data, e.g.
cd plotscripts
gnuplot ratio_bigheron
