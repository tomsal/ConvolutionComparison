//#include <stdio.h>
//#include <cstdlib>
#include <iostream>
#include <map>

#include <arrayfire.h>

#include <vigra/convolution.hxx>
#include <vigra/impex.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/random.hxx>

#include "boost/program_options.hpp"
#include "boost/timer/timer.hpp"


int main(int argc, char* argv[]){

  // --- Start command line parsing ---
  namespace po = boost::program_options;
  po::options_description po_desc("Options");
  po_desc.add_options()
    ("help","Print this help message.")
    ("N", po::value<int>(), "Amount of measurements taken. Default: 10")
    ("size", po::value<int>(), "Set image size. Default: 1024")
    ("sigma", po::value<float>(), "Set sigma for gaussianBlur. Default: 1.0")
    ("raw", po::value<bool>(), "Produce raw output (for plotting). Default: false")
    ("device", po::value<int>(), "Select GPU device number. Default: 0")
  ;

  po::variables_map options;
  po::store(po::parse_command_line(argc, argv, po_desc), options);
  po::notify(options);

  if(options.count("help")){
    std::cout << po_desc << "\n";
    return 1;
  }

  const int size = options.count("size") ? options["size"].as<int>() : 1024;
  const int N = options.count("N") ? options["N"].as<int>() : 10;
  const float sigma = options.count("sigma") ? options["sigma"].as<float>() : 1.0;
  const bool raw = options.count("raw") ? options["raw"].as<bool>() : false;
  const int device = options.count("device") ? options["device"].as<int>() : 0;
  // --- End command line parsing ---

  // --- Start create input image
  float* h_img = (float*) malloc(sizeof(float)*size*size); // random input image
  try{
    af::randu(size, size).host((void*)&h_img[0]);
  } catch (af::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    if(e.err() == AF_ERR_UNKNOWN)
      fprintf(stderr,"Maybe the nvidia driver is not loaded correctly.\n");
    return 1;
  }
  // --- End create input image

  // select GPU device
  af::setDevice(device);

  // create timer object
  boost::timer::auto_cpu_timer timer;

  // map for timings

  // create kernel for arrayfire that equals vigra kernel
  timer.start();
    const int ksize = int(sigma*3 + 0.5)*2 + 1;
    af::array gaussianKernel = af::gaussianKernel(ksize,ksize,sigma,sigma);
  timer.stop();
  float k_time = boost::lexical_cast<float>(timer.format(8,"%w"));

  std::cout << "Time: " << k_time << "\n";

  float* h_out = (float*) malloc(sizeof(float)*size*size); // convolution out
  std::cout << "Image size:" << size << "\n";

  return 0;
}
