#include <iostream>

#include <vigra/convolution.hxx>
#include <vigra/impex.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/random.hxx>

#define BILLION 1E9

using namespace vigra;

float mean(float* array, int k){
  float mean = 0;
  for(int i = 0; i < k; i++){
    mean+=array[i];
  }
  return (mean/k);
}

float max_array(float* arr, int length){
  float max = 0;
  for(int i = 0; i < length; i++){
    max = (arr[i] > max) ? arr[i] : max;
  }

  return max;
}

main(int argc, char ** argv){
  float sigma = 10.0;
  int h = 3072;
  int w = 3072;

   //vigra 
  char* in_filename = argv[1];
  ImageImportInfo imageInfo(in_filename);
  MultiArray<2, float> imageArray(Shape2(3072,3072));//imageInfo.shape());
  MultiArray<2, float> resultArray(Shape2(3072,3072));//imageInfo.shape());

  MersenneTwister random;
  //importImage(imageInfo, imageArray);
  for(int i = 0; i < 3072; i++)
    for(int j = 0; j < 3072; j++)
      imageArray(i,j) = random.uniform();

  //speed comparison

  float min = -1;
  int rep = 10;
  float dif, m;
  struct timespec start, end;  

  float* times = (float*)calloc(rep,sizeof(float));
  if (times == NULL) return -1;

  std::cout << "Convolution using Sigma:" << sigma <<std::endl;
  for(int i=1; i<=rep; i++){
    std::cout << "Iteration: " << i << "\n";
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    gaussianSmoothing(imageArray, resultArray, sigma);
    /*MultiArray<2, float> tmp(imageInfo.shape());
    Kernel1D <float> smooth, deriv;
    smooth.initGaussian(sigma);
    deriv.initGaussianDerivative(sigma, 2);
    separableConvolveX(imageArray,resultArray, deriv);
    //separableConvolveY(tmp,resultArray, smooth);
    //separableConvolveX(imageArray, tmp, smooth);
    separableConvolveY(imageArray,resultArray2, deriv);
    */
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    dif =  (end.tv_sec - start.tv_sec) + (float)(end.tv_nsec - start.tv_nsec )/ BILLION;
    min = ((min == -1.0) || (dif < min)) ? dif : min;
    *(times +i-1) = dif;
  }
  //print_array(times,rep);
  std::cout << "Convolution\n";
  std::cout << "mean:" << "\tvigra " << mean(times, rep) << std::endl;
  std::cout << "minimum\t:" << "\tvigra " << min << std::endl;
  //std::cout << "Faktor:\t" << mean(times,rep)/mean(times,rep) << std::endl;
  //printf("std for %i repitions: %e \n",rep, std(times, m, rep));*/

  return 0;
}
