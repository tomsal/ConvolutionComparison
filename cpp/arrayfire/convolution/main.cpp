/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include <cstdlib>
#include <arrayfire.h>

using namespace af;

// use static variables at file scope so timeit() wrapper functions
// can reference image/kernels
// image to convolve
static array img;

// 5x5 derivative with separable kernels
//static float h_dx[] = {1.f / 12, -8.f / 12, 0, 8.f / 12, -1.f / 12}; // five point stencil
//static float h_spread[] = {1.f / 5, 1.f / 5, 1.f / 5, 1.f / 5, 1.f / 5};
static float h_img[9437184];
static float h_out[9437184];
static array dx, spread;
static array kernel; // device kernels
static array full_out, dsep_out, hsep_out; // save output for value checks
static int N = 0;

// wrapper functions for timeit() below
static void full() { 
  img = array(3072,3072,h_img);
  full_out = convolve2(img, kernel);
  full_out.host((void*)&h_out[0]);
}

static void dsep() { 
  img = array(3072,3072,h_img);
  dsep_out = convolve(spread, dx, img);
  dsep_out.host((void*)&h_out[0]);
}

static void create_data(){
  for(int i = 0; i < 9437184; i++){
    h_img[i] = 1.f;
  }
}

static bool fail(array &left, array &right){
    return (max<float>(abs(left - right)) > 1e-6);
}

int main(int argc, char **argv){
    try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        //N = argc > 2 ? atoi(argv[2]) : 100;

        // kernel computation
        //dx = array(5, 1, h_dx); // 5x1 kernel
        dx = gaussianKernel(11,1,10.0,0.0); // 17x1 kernel
        kernel = gaussianKernel(61,61,10.0,10.0);
        
        //spread = array(1, 5, h_spread); // 1x5 kernel
        spread = gaussianKernel(1,11,0.0,10.0); // 1x17 kernel
        //kernel = matmul(dx, spread); // 5x5 kernel
        printf("create 1s image: %.5f seconds\n", timeit(create_data));

        // setup image and device copies of kernels
        timer::start();
        img = randu(3072, 3072);
        printf("Create random image:         %.5f seconds\n", timer::stop());
        printf("full 2D convolution (N = %d):         %.5f seconds\n", N, timeit(full));
        printf("separable, device pointers (N = %d):  %.5f seconds\n", N, timeit(dsep));
        // ensure values are all the same across versions
        //if (fail(full_out, dsep_out)) { throw af::exception("full != dsep"); }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
    }
    return 0;
}

