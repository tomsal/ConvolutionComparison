#include <stdio.h>
#include <cstdlib>
#include <arrayfire.h>

#include <vigra/convolution.hxx>
#include <vigra/impex.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/random.hxx>

#define SIZE 3072

// use static variables at file scope so timeit() wrapper functions
// can reference image/kernels
// image to convolve
static af::array img;

// 5x5 derivative with separable kernels
static float h_img[SIZE*SIZE];
static float h_out[SIZE*SIZE];
static af::array dx, spread;
static af::array kernel; // device kernels
static af::array full_out, dsep_out, hsep_out; // save output for value checks
static int N = 0;
const static float sigma = 10.0;
const int ksize = int(sigma*3 + 0.5)*2 + 1;

// vigra arrays
vigra::MultiArray<2, float> imageArray(vigra::Shape2(SIZE,SIZE));//imageInfo.shape());
vigra::MultiArray<2, float> resultArray(vigra::Shape2(SIZE,SIZE));//imageInfo.shape());

static void load_data_from_array(){
  for(int i = 0; i < SIZE; i++)
    for(int j = 0; j < SIZE; j++)
      imageArray(i,j) = h_img[i+j*SIZE];
}

static void vigra_gauss() {
  vigra::gaussianSmoothing(imageArray, resultArray, sigma);
}

// wrapper functions for timeit() below
static void af_full() { 
  img = af::array(SIZE,SIZE,h_img);
  full_out = convolve2(img, kernel);
  full_out.host((void*)&h_out[0]);
}

static void af_dsep() { 
  img = af::array(SIZE,SIZE,h_img);
  dsep_out = convolve(spread, dx, img);
  dsep_out.host((void*)&h_out[0]);
}

static void create_data(){
  img = af::randu(SIZE, SIZE);
  img.host((void*)&h_img[0]);
}

static void compare(){
  float d = 0;
  for(int i = ksize; i < SIZE-ksize; i++)
    for(int j = ksize; j < SIZE-ksize; j++)
      if((d = resultArray(i,j) - h_out[i+j*SIZE]) > 1e-6)
	printf("Difference for (%d,%d) = %f\n",i,j,d);
}

static bool fail(af::array &left, af::array &right){
    return (af::max<float>(abs(left - right)) > 1e-6);
}

int main(int argc, char **argv){
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        //N = argc > 2 ? atoi(argv[2]) : 100;

        // kernel computation
        //dx = af::array(5, 1, h_dx); // 5x1 kernel
        dx = af::gaussianKernel(11,1,10.0,0.0); // 17x1 kernel
        kernel = af::gaussianKernel(ksize,ksize,sigma,sigma);
        
        //spread = af::array(1, 5, h_spread); // 1x5 kernel
        spread = af::gaussianKernel(1,11,0.0,10.0); // 1x17 kernel
        //kernel = af::matmul(dx, spread); // 5x5 kernel
        printf("create 1s image: %.5f seconds\n", af::timeit(create_data));
	load_data_from_array();

        // setup image and device copies of kernels
        //af::timer::start();
        //img = af::randu(SIZE, SIZE);
        //printf("Create random image:         %.5f seconds\n", af::timer::stop());

        printf("af_full 2D convolution (N = %d):         %.5f seconds\n", N, af::timeit(af_full));
        printf("vigra_gauss 2D convolution (N = %d):         %.5f seconds\n", N, af::timeit(vigra_gauss));
	compare();

        //printf("separable, device pointers (N = %d):  %.5f seconds\n", N, af::timeit(af_dsep));
        // ensure values are all the same across versions
        //if (fail(full_out, dsep_out)) { throw af::exception("full != dsep"); }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}

