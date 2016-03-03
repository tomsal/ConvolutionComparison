#include <iostream>

#include <arrayfire.h>

#include <vigra/convolution.hxx>
#include <vigra/impex.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/random.hxx>

#include "boost/accumulators/accumulators.hpp"
#include "boost/accumulators/statistics/stats.hpp"
#include "boost/accumulators/statistics/mean.hpp"
#include "boost/accumulators/statistics/variance.hpp"
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
    ("raw", po::value<bool>()->zero_tokens(), "Produce raw output (for plotting). Default: false")
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

  if(!raw){
    std::cout << "--- Parameters ---\n";
    std::cout << "\tSize: " << size << "\n";
    std::cout << "\tN: " << N << "\n";
    std::cout << "\tSigma: " << sigma << "\n";
    std::cout << "\tDevice: " << device << "\n";
  }

  // select GPU device
  af::setDevice(device);

  // create timer object
  boost::timer::auto_cpu_timer timer;

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

  // load vigra image
  vigra::MultiArray<2, float> v_img(vigra::Shape2(size,size));
  for(int i = 0; i < size; i++)
    for(int j = 0; j < size; j++)
      v_img(i,j) = h_img[i+j*size];
  // vigra convolution result
  vigra::MultiArray<2, float> v_result(vigra::Shape2(size,size));

  // arrayfire arrays
  af::array d_img; // input image
  af::array d_result; // convolution result
  float* h_result = (float*) malloc(sizeof(float)*size*size); // af convolution result host

  // create kernel for arrayfire that equals vigra kernel
  timer.start();
    const int ksize = int(sigma*3 + 0.5)*2 + 1;
    af::array d_kernel = af::gaussianKernel(ksize,ksize,sigma,sigma);
  timer.stop();
  float t_kernel = boost::lexical_cast<float>(timer.format(8,"%w"));

  if(!raw)
    std::cout << "--- Starting measurements ---\n";

  // --- Start measurement loop ---
  namespace accus = boost::accumulators;
  accus::accumulator_set<float, accus::stats<accus::tag::variance> > acc_af_cpy_hd;
  accus::accumulator_set<float, accus::stats<accus::tag::variance> > acc_af_convolve;
  accus::accumulator_set<float, accus::stats<accus::tag::variance> > acc_af_cpy_dh;
  accus::accumulator_set<float, accus::stats<accus::tag::variance> > acc_vigra;
  for(int i = 0; i < N; ++i){
    // --- Start af measurements ---
    // Create device array, i.e. copy h_img to GPU memory
    timer.start();
      d_img = af::array(size,size,h_img);
      af::sync();
    timer.stop();
    acc_af_cpy_hd(boost::lexical_cast<float>(timer.format(8,"%w")));

    timer.start();
      d_result = af::convolve2(d_img,d_kernel);
      af::sync();
    timer.stop();
    acc_af_convolve(boost::lexical_cast<float>(timer.format(8,"%w")));

    timer.start();
      d_result.host((void*)&h_result[0]);
      af::sync();
    timer.stop();
    acc_af_cpy_dh(boost::lexical_cast<float>(timer.format(8,"%w")));
    // --- End af measurements ---

    // --- Start vigra measurements ---
    timer.start();
      vigra::gaussianSmoothing(v_img, v_result, sigma);
    timer.stop();
    acc_vigra(boost::lexical_cast<float>(timer.format(8,"%w")));
    // --- End vigra measurements ---
  }

  float af_hd_mean = accus::mean(acc_af_cpy_hd);
  float af_hd_variance = accus::variance(acc_af_cpy_hd);

  float af_convolve_mean = accus::mean(acc_af_convolve);
  float af_convolve_variance = accus::variance(acc_af_convolve);

  float af_dh_mean = accus::mean(acc_af_cpy_dh);
  float af_dh_variance = accus::variance(acc_af_cpy_dh);

  float af_total_mean = af_hd_mean + af_convolve_mean + af_dh_mean;
  float af_total_stddev = sqrt(af_hd_variance + af_convolve_variance + af_dh_variance);

  if(!raw){
    std::cout << "--- Results ---\n";
    std::cout << "\tMean copy h->d = " << af_hd_mean << " s\n";
    std::cout << "\tStandard deviation copy h->d = " << sqrt(af_hd_variance) << " s\n\n";
    std::cout << "\tMean convolve = " << af_convolve_mean << " s\n";
    std::cout << "\tStandard deviation convolve = " << sqrt(af_convolve_variance) << " s\n\n";
    std::cout << "\tMean copy d->h = " << af_dh_mean << " s\n";
    std::cout << "\tStandard deviation copy d->h = " << sqrt(af_dh_variance) << " s\n\n";

    std::cout << "\tArrayfire total mean = " << af_total_mean << " s\n\n";
    std::cout << "\tArrayfire total stddev = " << af_total_stddev << " s\n\n";
    
    std::cout << "\tMean vigra = " << accus::mean(acc_vigra) << " s\n";
    std::cout << "\tStandard deviation vigra = " << sqrt(accus::variance(acc_vigra)) << " s\n";
  }
  else{ // if(raw)
    if(N > 0){
      std::cout << size << ", " << sigma << ", "
		<< af_hd_mean << ", " << sqrt(af_hd_variance) << ", "
		<< af_convolve_mean << ", " << sqrt(af_convolve_variance) << ", "
		<< af_dh_mean << ", " << sqrt(af_dh_variance) << ", "
		<< af_total_mean << ", " << af_total_stddev << ", "
		<< accus::mean(acc_vigra) << ", " << sqrt(accus::variance(acc_vigra)) << "\n";
    }
    else{
      std::cout << "# size, sigma, af cpy h->d mean [s], af cpy h->d stddev [s],"
		   " af convolve mean [s], af convolve stddev [s],"
		   " af cpy d->h mean [s], af cpy d->h stddev [s],"
                   " af total mean [s], af total stddev [s],"
		   " vigra mean [s], vigra stddev [s]\n";
    }
  }

  return 0;
}
