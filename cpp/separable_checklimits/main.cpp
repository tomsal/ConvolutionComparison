#include <iostream>
#include <string>

#include <arrayfire.h>

//#include "boost/accumulators/accumulators.hpp"
//#include "boost/accumulators/statistics/stats.hpp"
//#include "boost/accumulators/statistics/mean.hpp"
//#include "boost/accumulators/statistics/variance.hpp"
#include "boost/program_options.hpp"
//#include "boost/timer/timer.hpp"


int main(int argc, char* argv[]){

  // --- Start command line parsing ---
  namespace po = boost::program_options;
  po::options_description po_desc("Options");
  po_desc.add_options()
    ("help","Print this help message.")
    ("error", po::value<bool>()->zero_tokens(), "Show only failed attemps/errors. Default: false")
    ("mode", po::value<std::string>(), "Set mode for 2D convolution, i.e. common or separable (2D or sep). Default: 2D")
    ("device", po::value<int>(), "Select GPU device number. Default: 0")
  ;

  po::variables_map options;
  po::store(po::parse_command_line(argc, argv, po_desc), options);
  po::notify(options);

  if(options.count("help")){
    std::cout << po_desc << "\n";
    return 1;
  }

  const bool error = options.count("error") ? options["error"].as<bool>() : false;
  const std::string mode = options.count("mode") ? options["mode"].as<std::string>() : "2D";
  const int device = options.count("device") ? options["device"].as<int>() : 0;
  // --- End command line parsing ---

  std::cout << "--- Parameters ---\n";
  std::cout << "\tError: " << error << "\n";
  std::cout << "\tMode: " << mode << "\n";
  std::cout << "\tDevice: " << device << "\n";

  // select GPU device
  af::setDevice(device);
  af::info();

  const int NSIZES = 7;
  int sizes[NSIZES] = {256, 512, 1024, 2048, 3072, 4096, 5120};
  const int NSIGMAS = 11;
  float sigmas[NSIGMAS] = {0.7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

  for(int i = 0; i < NSIZES; i++){
    af::array afimg = af::randu(sizes[i], sizes[i]);
    for(int j = 0; j < NSIGMAS; j++){
      const int ksize = 2*int(3*sigmas[j])+1;
      try{
	if(mode == "2D"){
          af::array afk = af::gaussianKernel(ksize,ksize,sigmas[j],sigmas[j]);
          af::array afres = af::convolve2(afimg, afk);
	}
	if(mode == "sep"){
          af::array afk = af::gaussianKernel(ksize,1,sigmas[j],0);
          af::array afres = af::convolve(afk, af::transpose(afk), afimg);
	}
	if(!error)
	  std::cout << "OK: size = " << sizes[i]
		    << " -- sigma = " << sigmas[j] << " -- kernel_size = " << ksize << "\n";
      } catch (af::exception& e) {
	std::cout << "ERROR: " << e.err() << " # size = " << sizes[i]
		  << " -- sigma = " << sigmas[j] << " -- kernel_size = " << ksize << "\n";
      }
    }
  }

  return 0;
}
