#include <arrayfire.h>

int main(int argc, char* argv[]){
  int device = argc > 1 ? atoi(argv[1]) : 0;

  af::info();
  af::setDevice(device);
  af::info();

  af::array img = af::randu(3072,3072);
  af::array kernel = af::gaussianKernel(61,61,10.0,10.0);
  af::array out = convolve(img, kernel);

  af::info();

  return 0;
}
