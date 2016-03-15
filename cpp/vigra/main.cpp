//#include <vigra/convolution.hxx>
//#include <vigra/impex.hxx>
//#include <vigra/imageinfo.hxx>
//#include <vigra/stdimage.hxx>
//#include <vigra/multi_array.hxx>
//#include <vigra/hdf5impex.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/multi_blockwise.hxx>
//#include <vigra/random.hxx>

// uncomment to remove boost dependencies, i.e. time measurements
// #define NOBOOSTTIMER

#ifndef NOBOOSTTIMER
#include "boost/timer/timer.hpp"
#endif

int main(int argc, char** argv){
  int size = 4096;
  float sigma = 10.0;

  // load vigra image
  vigra::MultiArray<2, float> v_img(vigra::Shape2(size,size));
  for(int i = 0; i < size; i++)
    for(int j = 0; j < size; j++)
      v_img(i,j) = float(i*j); //h_img[i+j*size];

  // vigra convolution result
  vigra::MultiArray<2, float> v_result(vigra::Shape2(size,size));

  vigra::BlockwiseConvolutionOptions<2> opt;
  opt.innerScale(sigma);
  opt.setNumThreads(8);

#ifndef NOBOOSTTIMER
  boost::timer::auto_cpu_timer timer;
  timer.start();
#endif

  vigra::gaussianSmoothMultiArray(v_img, v_result, opt);

#ifndef NOBOOSTTIMER
  timer.elapsed();
#endif

  return 0;
}

